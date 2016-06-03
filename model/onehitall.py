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
        data = extractData(tsv, path, use_reach)

        # Skip this paper if it has less than 10 positive annotations
        if len([d for d in data if d.label == 1]) < 20:
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


                negatives = np.random.choice(negatives, size=size if size <= len(negatives) else len(negatives), replace=False)

                augmented = data + list(negatives)
            pos += len(data)
            neg += len(negatives)
            # print '%s Positives:%i Negatives:%i Total:%i' % (pmcid, len(data), len(negatives), len(data)+len(negatives))


        for ix, datum in enumerate(augmented):
            accumulated_data.append(datum)
            vector = classifier.createFeatures(datum, tsv, annotationData)
            features[datum] = vector


        for datum in augmented:
            labels[datum] = datum.label

        #accumulator += len(augmented)

    # print
    # print 'Positives: %i Negatives:%i Total:%i' % (pos, neg, pos + neg)
    # print

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

def crossval(paths, annDir, eval_type, use_reach, relabeling):
    ''' Puts all together '''

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

    print "Using %i papers" % len(groups2)

    # Make it a numpy array to index it more easily
    data = np.asarray(data)

    dv = DictVectorizer()
    dv.fit(features.values())

    # Build a feature vector and attach it to each datum
    vectors = {k:dv.transform(v) for k, v in features.iteritems()}


    c_results, p_results = [], []

    # Do the "Cross-validation" only on those papers that have more than N papers
    for path in groups.keys():

        training_paths = paths - {path}

        X_train, X_test = [], []
        y_train, y_test = [], []
        data_test = []

        for datum in data:
            if datum.namespace in training_paths:
                X_train.append(vectors[datum])
                y_train.append(labels[datum])
            else:
                X_test.append(vectors[datum])
                y_test.append(labels[datum])
                data_test.append(datum)


        model_pred = machine_learning(vstack(X_train), y_train, vstack(X_test), y_test)

        policy_results = ClassificationResults("Model %s" % path, y_test, model_pred)

        policy_pred = policy(np.asarray(data_test))
        policy_result = ClassificationResults("Policy %s" % path, y_test, policy_pred)

        c_results.append(policy_results)
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

    use_reach = True
    relabeling = True
    ev = EVAL1

    if use_reach:
        print "Using REACH's data"

    model_results, policy_results = crossval(paths, annDir, eval_type=ev, use_reach = use_reach, relabeling=relabeling)

    macro_results, micro_results = MacroAverage("Macro model", model_results), MicroAverage("Micro model", model_results)
    macro_policy, micro_policy = MacroAverage("Macro policy", model_results), MicroAverage("Micro policy", model_results)
    # t test
    # t = ttest_1samp(f1_diffs, 0)
    #
    # print "Mean diff:  %f" % f1_diffs.mean()
    # print "p-value: %f" % t.pvalue

    # testing_f1_diff = testing(paths, annDir, testing_ids, eval_type=ev, use_reach = use_reach)
    # print "testing F1 diff: %f" % testing_f1_diff
