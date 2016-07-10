''' This is the file that implements the vector aggregation of the features for each event mention and context type
    It does cross validation over all the documents too '''


from __future__ import division
import csv, glob, sys, os, itertools, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parsing
from collections import defaultdict
from sklearn.linear_model import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import *
from sklearn.preprocessing import normalize
from sklearn import cross_validation, metrics
from sklearn.feature_extraction import DictVectorizer
from parsing import *
from model import *
from scipy.stats import bernoulli, ttest_1samp
from unbalanced_dataset import *
from sklearn.metrics import f1_score
from scipy.sparse import vstack
from classification_results import *
import classifier
import os
from joblib import Parallel, delayed

def in_neighborhood(datum, intervals):
    ''' Used to filter a datum if it's not within the neighborhood of (min, max) '''

    for interval in intervals:
        minIx, maxIx = interval
        if datum.ctxIx >= minIx and datum.ctxIx <= maxIx:
            return True

    return False

def parse_data(paths, annDir, use_reach, relabeling, skip_amount=10):
    ''' Gets the Datum objects *before* extracting features '''

    # Read everything
    accumulated_data = {}

    #accumulator = 0
    total_pos, total_neg = 0, 0
    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        tsv = parseTSV(path)
        annotationData = extractAnnotationData(pmcid, annDir)
        data = extractData(tsv, path, annotationData, use_reach)

        # Attach the tsv and the annotation data to each datum
        for datum in data:
            datum.tsv = tsv
            datum.annotationData = annotationData

        # Skip this paper if it has less than 10 positive annotations
        if len([d for d in data if d.label == 1]) < skip_amount:
            continue


        augmented = data

        # We need negative examples
        negatives = generateNegativesFromNER(data, annotationData, relabeling)

        if len(negatives) > 0:

            size = len(negatives)
            negatives = list(np.random.choice(negatives, size=size if size <= len(negatives) else len(negatives), replace=False))

            # Attach the tsv and the annotation data to each datum
            for datum in negatives:
                datum.tsv = tsv
                datum.annotationData = annotationData

            augmented += negatives

        pos = sum([1 for a in augmented if a.label == 1])
        neg = sum([1 for a in augmented if a.label == 0])

        total_pos += pos
        total_neg += neg

        print '%s Positives: %i Negatives: %i Total: %i' % (pmcid, pos, neg, pos+neg)
        accumulated_data[pmcid] = augmented

    print
    print 'All papers. Positives: %i Negatives: %i Total: %i' % (total_pos, total_neg, total_pos+total_neg)
    print
    return accumulated_data

def create_features(data):
    ''' Creates a feature vector and attaches it to the datum object '''

    for datum in data:
        datum.features = classifier.createFeatures(datum, datum.tsv, datum.annotationData)

def vectorize_data(data):
    ''' Creates Numpy feature vectors out of datum objects '''

    cv = DictVectorizer()
    X = cv.fit_transform([d.features for d in data])
    # Normalize the matrix
    normalize(X, norm='max', copy=False)

    for i, datum in enumerate(data):
        datum.vector = X[i, :]

def crossval_baseline(folds, conservative_eval):
    # This is essentially a "one-hit-all" evaluation of policy 4

    results = {}

    for fold_name, data in folds:
        predictions = {}
        truths = {}
        
        for datum in data:
            prediction = classifier.policy(datum, 3)
            truth = datum.label
            key = (datum.evt, datum.ctxGrounded)

            if key in predictions:
                # If the prediction if false, but the value is true,
                # lets override it. Otherwise the result is the same
                if predictions[key] == 0:
                    predictions[key] = prediction

                if truths[key] == 0:
                    truths[key] = truth
            else:
                # Store a result anyway
                predictions[key] = prediction
                truths[key] = truth

        keys = predictions.keys()
        result = ClassificationResults(fold_name, [truths[k] for k in keys], [predictions[k] for k in keys], [k for k in keys])
        results[fold_name] = result

    return results


def add_vectors(data, average=False):
    ''' Adds all the vectors from an event mention to a context type
        Optionally averages them '''

    def key(datum): return datum.evt, datum.ctxGrounded

    data = sorted(data, key=key)
    groups = it.groupby(data, key)

    new_points, vectors, labels = [], [], []

    # Group them by event mention and ctx type
    for k, v in groups:
        new_points.append(k)
        local_vectors, local_labels = [], []
        for datum in v:
            local_vectors.append(datum.vector)
            local_labels.append(datum.label)

        # Add them together
        vector = sum(local_vectors)
        # Average them if specified
        if average:
            vector /= len(local_vectors)

        vectors.append(vector)

        # Get the label
        label = 1 if sum(local_labels) >= 1 else 0
        labels.append(label)

    return new_points, vstack(vectors), np.asarray(labels)


def train_eval_model(name, X_train, X_test, y_train, y_test, point_labels):
    ''' Configure the selected algorithm here and return a ClassificationResults object '''

    # Edit the algorithm here
    # model = Perceptron(penalty='l2')
    model = LogisticRegression(penalty='l2', C=10)
    #########################

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return ClassificationResults(name, y_test, predictions, point_labels)


def crossval_model(folds, conservative_eval, limit_training, balance_dataset):

    aggregated_data = {}
    for fold_name, data in folds.iteritems():
        aggregated_data[fold_name] = add_vectors(data)


    results = {}

    for fold_name in folds:
        point_labels, X_test, y_test = aggregated_data[fold_name]


        X_train = vstack([aggregated_data[f][1] for f in aggregated_data if f != fold_name])
        y_train = np.concatenate([aggregated_data[f][2] for f in aggregated_data if f != fold_name])

        print "Testing on %s ..." % fold_name
        results[fold_name] = train_eval_model(fold_name, X_train, X_test, y_train, y_test, point_labels)

    return results


# Entry point
if __name__ == "__main__":

    # Curated TSV directory
    directory = sys.argv[1]
    # Reach's annotation directory
    annDir = sys.argv[2]

    paths = glob.glob(os.path.join(directory, '*.tsv'))

    # CONFIG ###
    use_reach = True # Use reach's context extractions to extend the data set
    relabeling = parsing.RELABEL # Relabel alternative examples if they have the same type as one of Xia's choices for it's event
    conservative_eval = False # Only evaluate over Xia's annotations
    limit_training = False # Only use the training examples that are in the neighborhood of the golden annotations
    # This is been deprecated, as we use aggregation now
    # one_hit_all = True # If one datum is classified positively, all data with the same context grounded id are postclassified as positive
    balance_dataset = True # Use a 1:K ratio of positive to negative examples during training
    ############

    print "Parsing data"
    cv_folds = parse_data(paths, annDir, use_reach = use_reach, relabeling=relabeling)

    print "Extracting features"
    create_features(it.chain(*cv_folds.values()))

    print "Baseline cross validation"
    policy_results = crossval_baseline(cv_folds.iteritems(), conservative_eval=conservative_eval)

    print "Vectorizing features"
    vectorize_data(list(it.chain(*cv_folds.values())))

    print "Machine Learning cross validation"
    model_results = crossval_model(cv_folds, conservative_eval=conservative_eval,\
                                            limit_training=limit_training, balance_dataset=balance_dataset)

    macro_model, micro_model = MacroAverage("Macro model", model_results.values()), MicroAverage("Micro model", model_results.values())
    macro_policy, micro_policy = MacroAverage("Macro policy", policy_results.values()), MicroAverage("Micro policy", policy_results.values())

    os.system('say "your program has finished"')
