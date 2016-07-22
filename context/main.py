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
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.preprocessing import normalize
from sklearn import cross_validation, metrics
from sklearn.feature_extraction import *
from parsing import *
from scipy.stats import bernoulli, ttest_1samp
from unbalanced_dataset import *
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFE
from scipy.sparse import vstack, hstack
from classification_results import *
from utils import *
from context.features import *
import os
from context import *
from context.utils import *

# Set the random seed for replicability
np.random.seed(0)

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

def vectorize_data(data):
    ''' Creates scipy sparse feature vectors out of datum objects '''

    cv, cv2 = DictVectorizer(), DictVectorizer()
    X, X2 = cv.fit_transform([d.features for d in data]), cv2.fit_transform([d.doc_features for d in data])
    # Normalize the matrix
    normalize(X, norm='max', copy=False)
    normalize(X2, norm='max', copy=False)

    for i, datum in enumerate(data):
        datum.vector = hstack([X[i, :], X2[i, :]])

    # Return the dict vectorizers to transform back the vectors into dictionaries
    return cv, cv2

def crossval_baseline(folds):
    ''' Cross validation of the deterministic classifier as a baseline '''
    # This is essentially a "one-hit-all" evaluation of policy 4 in reach

    results = {}

    # Each document is a fold
    for fold_name, data in folds:
        predictions = {}
        truths = {}

        for datum in data:
            prediction = policy(datum, 3)
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

    def key(datum): return datum.namespace, datum.evt, datum.ctxGrounded

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
    ''' Configure the selected algorithm here and return a ClassificationResults object and a feature mask '''

    verbose = False
    # Edit the algorithm here
    # model = Perceptron(penalty='l2')
    algorithm = LogisticRegression(penalty='l2', C=.1)
    # model = SVC(verbose=verbose, kernel='linear', C=1)
    # algorithm = SVC(verbose=verbose, kernel='rbf', C=1)
    # model = SVC(verbose=verbose, kernel='poly', degree=3, C=50)
    # model = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=2)
    # model = RandomForestClassifier(n_estimators=10, max_depth=2)
    #########################

    algorithm.fit(X_train, y_train)
    predictions = algorithm.predict(X_test)
    coefficients = algorithm.coef_

    # Recursive feature selection
    # selector = RFE(algorithm, step=25, verbose=1)
    # selector = selector.fit(X_train, y_train)
    # print selector.support_.astype(int).sum()
    # Dimension-reduced data
    # mask = selector.support_
    # X_train, X_test = X_train[:, mask], X_test[:, mask]
    # Ensemble
    # ensemble = BaggingClassifier(LogisticRegression(penalty='l2', C=5), n_estimators=50, max_samples=1.0, verbose=2, n_jobs=3)
    # ensemble.fit(X_train, y_train)
    # coefficients = np.asarray([e.coef_[0] for e in ensemble.estimators_]).mean()
    # predictions = ensemble.predict(X_test)

    # Ensemble
    # ensemble = BaggingClassifier(LogisticRegression(penalty='l2', C=5), n_estimators=50, max_samples=.70, verbose=2, n_jobs=-1)
    # ensemble.fit(X_train, y_train)
    # coefficients = np.asarray([e.coef_[0] for e in ensemble.estimators_]).mean()
    # predictions = ensemble.predict(X_test)

    # coefficients = selector.estimator_.coef_[0]
    # predictions = selector.predict(X_test)


    # TODO: Maybe remove this
    mask = np.ones(X_train.shape[1]).astype(bool)
    return ClassificationResults(name, y_test, predictions, point_labels), mask, coefficients

def crossval_model(folds, limit_training, balance_dataset):
    ''' Cross validation of the context classifier '''

    # Add the vectors together
    aggregated_data = {}
    for fold_name, data in folds.iteritems():
        aggregated_data[fold_name] = add_vectors(data, False)


    # We want to keep track of these values
    results, masks, coeffictients = {}, {}, {}

    tests = []

    # Each document is a fold
    for fold_name in folds:
        # Fetch the data from the testing document
        point_labels, X_test, y_test = aggregated_data[fold_name]
        # Store the text matrix of this fold
        tests.append(X_test)

        # Add together all the training data. We use vstack from scipy because of the sparse matrices
        X_train = vstack([aggregated_data[f][1] for f in aggregated_data if f != fold_name])
        y_train = np.concatenate([aggregated_data[f][2] for f in aggregated_data if f != fold_name])


        if balance_dataset:
            # Downsample negatives
            k = 3
            neg_ix = np.where(y_train == 0)[0]
            pos_ix = np.where(y_train == 1)[0]
            subsampled_negs = np.random.choice(neg_ix, size=int(len(pos_ix)*k), replace=False)
            new_ix = np.concatenate([subsampled_negs, pos_ix])
            # np.random.shuffle(new_ix)
            X_train = X_train[new_ix, :]
            y_train = y_train[new_ix]

            # # Upsample positives
            # l = 1.5
            # pos_ix = np.where(y_train == 1)[0]
            # neg_ix = np.where(y_train == 0)[0]
            # upsampled_pos = np.random.choice(pos_ix, size=int(len(pos_ix)*l), replace=True)
            # new_ix = np.concatenate([upsampled_pos, neg_ix])
            # np.random.shuffle(new_ix)
            # X_train = X_train[new_ix, :]
            # y_train = y_train[new_ix]


        print "Testing on %s ..." % fold_name
        print "%i positive - %i negative" % (y_train.sum(), len(y_train)-y_train.sum())
        # Do training and testing of this fold
        results[fold_name], masks[fold_name], coeffictients[fold_name] = train_eval_model(fold_name, X_train, X_test, y_train, y_test, point_labels)

    # Return the data structures
    return results, masks, coeffictients, vstack(tests)


# Entry point
if __name__ == "__main__":

    # Curated TSV directory
    directory = sys.argv[1]
    # Reach's annotation directory
    annDir = sys.argv[2]

    # Get the paths of the curated tsvs
    paths = glob.glob(os.path.join(directory, '*.tsv'))

    # CONFIG ###
    use_reach = True # Use reach's context extractions to extend the data set
    relabeling = parsing.RELABEL # Relabel alternative examples if they have the same type as one of Xia's choices for it's event
    limit_training = False # Only use the training examples that are in the neighborhood of the golden annotations
    # This is been deprecated, as we use aggregation now
    # one_hit_all = True # If one datum is classified positively, all data with the same context grounded id are postclassified as positive
    balance_dataset = True # Use a 1:K ratio of positive to negative examples during training
    ############

    print "Parsing data"
    cv_folds = parse_data(paths, annDir, use_reach = use_reach, relabeling=relabeling)

    print "Extracting features"
    create_features(it.chain(*cv_folds.values()))

    # print "Baseline cross validation"
    policy_results = crossval_baseline(cv_folds.iteritems())

    print "Vectorizing features"
    vectorizer, vectorizer2 = vectorize_data(list(it.chain(*cv_folds.values())))

    print "# of features: %i" % cv_folds.values()[0][0].vector.shape[1]
    print "Machine Learning cross validation"
    model_results, masks, coefficients, X  = crossval_model(cv_folds,\
                            limit_training=limit_training, balance_dataset=balance_dataset)

    # Build macro and micro averages
    macro_model, micro_model = MacroAverage("Macro model", model_results.values()), MicroAverage("Micro model", model_results.values())
    macro_policy, micro_policy = MacroAverage("Macro policy", policy_results.values()), MicroAverage("Micro policy", policy_results.values())

    # Comment this out if you're not running MacOS
    os.system('say "your program has finished"')
