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
import classifier

np.random.seed(0)

def parse_data(paths, annDir, use_reach, testing=False):
    ''' Reads and process stuff '''

    # Read everything
    accumulated_data = []
    vectors = []
    labels = []

    hashes = {}
    accumulator = 0
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
            negatives = generateNegativesFromNER(data, annotationData)

            if len(negatives) > 0:
                # Randomly pick a subset of negatives
                if not testing:
                    size = len(negatives)
                    # data = copy_positives(data, len(negatives)//len(data))
                else:
                    size = len(negatives)

                # else:
                #     size = len(negatives)

                negatives = np.random.choice(negatives, size=size if size <= len(negatives) else len(negatives), replace=False)
                augmented = data + list(negatives)
            pos += len(data)
            neg += len(negatives)
            # print '%s Positives:%i Negatives:%i Total:%i' % (pmcid, len(data), len(negatives), len(data)+len(negatives))


        for ix, datum in enumerate(augmented):
            accumulated_data.append(datum)
            vector = classifier.createFeatures(datum, tsv, annotationData)
            vectors.append(vector)
            # Generate clusters for type 2 eval
            hashes[accumulator+ix] = hash(datum)

        labels += [datum.label for datum in augmented]
        accumulator += len(augmented)

    print
    # print 'Positives: %i Negatives:%i Total:%i' % (pos, neg, pos + neg)
    print

    return labels, vectors, hashes, accumulated_data

@np.vectorize
def policy(datum):
    return classifier.policy(datum, 3)

def machine_learning(X, y, data, train_ix, test_ix):
    X_train, X_test = X[train_ix], X[test_ix]
    y_train, y_test = y[train_ix], y[test_ix]
    data_test = data[test_ix]

    y_policy = policy(data_test)
    policy_f1 = f1_score(y_test, y_policy)

    normalize(X_train, norm='l2', copy=False)
    normalize(X_test, norm='l2', copy=False)

    lr = Perceptron(penalty='l2')

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    model_f1 = f1_score(y_test, y_pred)

    return policy_f1, model_f1

def testing(paths, annDir, testing_ids, eval_type, use_reach):
    ''' Puts all together for measuring the test set '''

    print "Parsing data"
    paths = set(paths)
    labels, vectors, hashes, data = parse_data(paths, annDir, use_reach)

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
    dv.fit(vectors)

    X = dv.transform(vectors)
    y = np.asarray(labels)

    indices = set(range(len(data)))

    test_ix = set(it.chain(*[groups[p] for p in testing_ids if p in groups]))
    train_ix = list(indices - test_ix)
    test_ix = list(test_ix)

    policy_f1, model_f1 = machine_learning(X, y, data, train_ix, test_ix)

    f1_diff = model_f1 - policy_f1

    return f1_diff

def crossval(paths, annDir, eval_type, use_reach):
    ''' Puts all together '''

    print "Parsing data"
    paths = set(paths)
    labels, vectors, hashes, data = parse_data(paths, annDir, use_reach)

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
    dv.fit(vectors)

    X = dv.transform(vectors)
    y = np.asarray(labels)

    f1_diffs = []
    model_f1s = {}

    indices = set(range(len(data)))
    # Do the "Cross-validation" only on those papers that have more than N papers
    for path in groups.keys():

        others = paths - {path}
        test_ix = set(groups[path])
        train_ix = list(indices - test_ix)
        test_ix = list(test_ix)

        policy_f1, model_f1 = machine_learning(X, y, data, train_ix, test_ix)

        f1_diffs.append(model_f1 - policy_f1)
        model_f1s[path] = model_f1

    return pd.Series(f1_diffs), model_f1s


# Entry point
if __name__ == "__main__":
    directory = sys.argv[1]
    annDir = sys.argv[2]
    testing_ids = {'%s/%s.tsv' % (directory, s[:-1]) for s in open(sys.argv[3])}

    paths = glob.glob(os.path.join(directory, '*.tsv'))
    #paths = ['/Users/enoriega/Dropbox/Context Annotations/curated tsv/PMC2063868_E.tsv']

    use_reach = True
    ev = EVAL1

    if use_reach:
        print "Using REACH's data"

    f1_diffs, model_f1s = crossval(paths, annDir, eval_type=ev, use_reach = use_reach)
    t = ttest_1samp(f1_diffs, 0)

    print "Individual F1 scores"
    for k, v in model_f1s.iteritems():
        print '%s: %f' % (k, v)

    print "Mean diff:  %f" % f1_diffs.mean()
    print "p-value: %f" % t.pvalue

    testing_f1_diff = testing(paths, annDir, testing_ids, eval_type=ev, use_reach = use_reach)
    print "testing F1 diff: %f" % testing_f1_diff
