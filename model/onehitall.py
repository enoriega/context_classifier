''' Paired t-test '''

from __future__ import division
import csv, glob, sys, os, itertools, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

np.random.seed(0)

def parse_data(paths, annDir, use_reach, relabeling, testing=False):
    ''' Reads and process stuff '''

    # Read everything
    accumulated_data = []
    features = {}
    labels = {}

    hashes = {}
    #accumulator = 0
    pos, neg = 0, 0
    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        tsv = parseTSV(path)
        annotationData = extractAnnotationData(pmcid, annDir)
        data = extractData(tsv, path, annotationData, use_reach)

        # Skip this paper if it has less than 10 positive annotations
        if len([d for d in data if d.label == 1]) < 1:
            continue

        annotationData = extractAnnotationData(pmcid, annDir)
        augmented = data

        if use_reach:
            # We need negative examples
            negatives = generateNegativesFromNER(data, annotationData, relabeling)

            if len(negatives) > 0:
                # Randomly pick a subset of negatives
                if not testing:
                    size = len(negatives)
                    # data = copy_positives(data, len(negatives)//len(data))
                else:
                    size = len(negatives)


                negatives = list(np.random.choice(negatives, size=size if size <= len(negatives) else len(negatives), replace=False))

                augmented = data + negatives
            pos += len(data)
            neg += len(negatives)
            # print '%s Positives:%i Negatives:%i Total:%i' % (pmcid, len(data), len(negatives), len(data)+len(negatives))


        for ix, datum in enumerate(augmented):
            accumulated_data.append(datum)
            vector = classifier.createFeatures(datum, tsv, annotationData)
            features[datum] = vector


        for datum in augmented:
            labels[datum] = datum.label

    return labels, features, accumulated_data

@np.vectorize
def policy(datum):
    return classifier.policy(datum, 3)

def machine_learning(X_train, y_train, X_test, y_test):

    # y_policy = policy(data_test)
    # policy_f1 = f1_score(y_test, y_policy)
    normalize(X_train, norm='l2', copy=False)
    normalize(X_test, norm='l2', copy=False)

    lr = Perceptron(penalty='l2')

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    return y_pred

def testing(paths, annDir, testing_ids, eval_type, use_reach):
    ''' Puts all together for measuring the test set '''

    print "Parsing data"
    paths = set(paths)
    labels, features, hashes, data = parse_data(paths, annDir, use_reach)

    # Group indexes by paper id
    groups = {p:[] for p in paths}

    for i, d in enumerate(data):
        groups[d.namespace].append(i)

    # Hack!!
    groups2 = {}
    for k, v in groups.iteritems():
        if len(v) != 0:
            groups2[k] = v

    groups = groups2

    print "Using %i papers" % len(groups2)

    # Make it a numpy array to index it more easily
    data = np.asarray(data)

    dv = DictVectorizer()
    dv.fit(features)

    X = dv.transform(features)
    y = np.asarray(labels)

    indices = set(range(len(data)))

    test_ix = set(it.chain(*[groups[p] for p in testing_ids if p in groups]))
    train_ix = list(indices - test_ix)
    test_ix = list(test_ix)

    policy_f1, model_f1 = machine_learning(X, y, data, train_ix, test_ix)

    f1_diff = model_f1 - policy_f1

    return f1_diff

def crossval(paths, annDir, eval_type, use_reach, relabeling, conservative_eval, limit_training, balance_dataset):
    ''' Puts all together '''

    def in_neighborhood(datum, intervals):
        ''' Used to filter a datum if it's not within the neighborhood of (min, max) '''

        for interval in intervals:
            minIx, maxIx = interval
            if datum.ctxIx >= minIx and datum.ctxIx <= maxIx:
                return True

        return False

    print "Parsing data"
    paths = set(paths)
    labels, features, data = parse_data(paths, annDir, use_reach, relabeling)

    # Group indexes by paper id
    groups = {p:[] for p in paths}

    for i, d in enumerate(data):
        groups[d.namespace].append(i)

    # Hack!!
    groups2 = {}
    for k, v in groups.iteritems():
        if len(v) != 0:
            groups2[k] = v

    groups = groups2

    print
    print "Cross-validation"
    print "Using %i papers" % len(groups2)
    if relabeling: print "Doing relabeling"
    if conservative_eval: print "Doing conservative evaluation"
    if limit_training: print "Limiting training data range"
    if balance_dataset: print "Balancing data set during training"
    if one_hit_all: print "One-hit-all"
    print "Total golden data: %i\tTotal expanded data: %i" % (len([d for d in data if d.golden]), len([d for d in data if not d.golden]))
    print

    # Only use the training data
    if limit_training:
        # Compute the range of the annotations
        intervals = defaultdict(list)
        k = 2
        for datum in data:
            if datum.golden:
                intervals[datum.namespace].append((datum.ctxIx-k, datum.ctxIx+k))

    # Make it a numpy array to index it more easily
    data = np.asarray(data)

    dv = DictVectorizer()
    dv.fit(features.values())

    # Build a feature vector and attach it to each datum
    vectors = {k:dv.transform(v) for k, v in features.iteritems()}


    c_results, p_results = [], []

    # Do the "Cross-validation" only on those papers that have more than N papers
    for ix, path in enumerate(groups.keys()):

        print "Fold: %i" % (ix+1)

        training_paths = paths - {path}

        X_train, X_test = [], []
        y_train, y_test = [], []
        data_train, data_test = [], []

        for datum in data:
            if datum.namespace in training_paths:

                if limit_training:
                    if not in_neighborhood(datum, intervals[datum.namespace]):
                        continue

                X_train.append(vectors[datum])
                y_train.append(labels[datum])
                data_train.append(datum)


        for datum in data:
            if datum.namespace not in training_paths:
                if conservative_eval:
                    if not datum.golden:
                        continue

                X_test.append(vectors[datum])
                y_test.append(labels[datum])
                data_test.append(datum)

        # Balance the dataset if necessary
        if balance_dataset:
            train_positive, train_negative = [], []
            for datum in data_train:
                if datum.label == 1:
                    train_positive.append(datum)
                else:
                    train_negative.append(datum)

            k = 4# Ratio of negatives per positives for balancing
            size = len(train_positive)*k
            if size < len(train_negative):
                balanced_negatives = np.random.choice(train_negative, size, replace=False).tolist()
            else:
                balanced_negatives = train_negative

            data_train = train_positive + balanced_negatives

            X_train = [vectors[datum] for datum in data_train]
            y_train = [labels[datum] for datum in data_train]

        p = len([d for d in data_train if d.label == 1])
        n = len([d for d in data_train if d.label == 0])
        r = n/float(p)
        print path
        print "Training data: %i positives\t%i negatives\t%f N:P ratio" % (p, n, r)
        p = len([d for d in data_test if d.label == 1])
        n = len([d for d in data_test if d.label == 0])
        r = n/float(p)
        print "Testing data: %i positives\t%i negatives\t%f N:P ratio" % (p, n, r)


        model_pred = machine_learning(vstack(X_train), y_train, vstack(X_test), y_test)
        policy_pred = policy(np.asarray(data_test))

        # One-hit-all approach
        if one_hit_all:
            ctx_types = list({d.ctxGrounded for d in data_test})
            ctx_types.sort()
            local_events = list({d.evt for d in data_test})
            local_events.sort

            predicted_bag = set()
            for datum, prediction in it.izip(data_test, model_pred):
                if prediction == 1:
                    predicted_bag.add((datum.evt, datum.ctxGrounded))

            policy_bag = set()
            for datum, prediction in it.izip(data_test, policy_pred):
                if prediction == 1:
                    policy_bag.add((datum.evt, datum.ctxGrounded))

            truth_bag = set()
            for datum, prediction in it.izip(data_test, y_test):
                if prediction == 1:
                    truth_bag.add((datum.evt, datum.ctxGrounded))

            new_model_pred, new_policy_pred, new_truth = [], [], []
            for evt in local_events:
                for ctx in ctx_types:
                    if (evt, ctx) in predicted_bag:
                        new_model_pred.append(1)
                    else:
                        new_model_pred.append(0)

                    if (evt, ctx) in policy_bag:
                        new_policy_pred.append(1)
                    else:
                        new_policy_pred.append(0)

                    if (evt, ctx) in truth_bag:
                        new_truth.append(1)
                    else:
                        new_truth.append(0)

        y_test = new_truth
        model_pred = new_model_pred
        policy_pred = new_policy_pred
        ######################

        model_results = ClassificationResults("Model %s" % path, y_test, model_pred)
        policy_result = ClassificationResults("Policy %s" % path, y_test, policy_pred)
        print "Model scores: %s" % model_results
        print "Policy scores %s" % policy_result
        print

        c_results.append(model_results)
        p_results.append(policy_result)


    #return pd.Series(f1_diffs), model_f1s
    return c_results, p_results


# Entry point
if __name__ == "__main__":
    directory = sys.argv[1]
    annDir = sys.argv[2]
    testing_ids = {'%s/%s.tsv' % (directory, s[:-1]) for s in open(sys.argv[3])}

    paths = glob.glob(os.path.join(directory, '*.tsv'))
    #paths = ['/Users/enoriega/Dropbox/Context Annotations/curated tsv/PMC2063868_E.tsv']

    use_reach = True # Use reach's context extractions to extend the data set
    relabeling = RELABEL # Relabel alternative examples if they have the same type as one of Xia's choices for it's event
    conservative_eval = False # Only evaluate over Xia's annotations
    limit_training = False # Only use the training examples that are in the neighborhood of the golden annotations
    one_hit_all = True # If one datum is classified positively, all data with the same context grounded id are postclassified as positive
    balance_dataset = True # Use a 1:K ratio of positive to negative examples during training

    ev = EVAL1

    if use_reach:
        print "Using REACH's data"

    model_results, policy_results = crossval(paths, annDir, eval_type=ev, use_reach = use_reach,\
                                            relabeling=relabeling, conservative_eval=conservative_eval,\
                                            limit_training=limit_training, balance_dataset=balance_dataset)

    macro_results, micro_results = MacroAverage("Macro model", model_results), MicroAverage("Micro model", model_results)
    macro_policy, micro_policy = MacroAverage("Macro policy", policy_results), MicroAverage("Micro policy", policy_results)

    os.system('say "your program has finished"')
    # t test
    # t = ttest_1samp(f1_diffs, 0)
    #
    # print "Mean diff:  %f" % f1_diffs.mean()
    # print "p-value: %f" % t.pvalue

    # testing_f1_diff = testing(paths, annDir, testing_ids, eval_type=ev, use_reach = use_reach)
    # print "testing F1 diff: %f" % testing_f1_diff
