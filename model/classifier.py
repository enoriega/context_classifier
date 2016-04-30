from __future__ import division
import csv, glob, sys, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn import cross_validation, metrics
from sklearn.feature_extraction import DictVectorizer
from parsing import *
from model import *

np.random.seed(0)

def createFeatures(datum, otherData, tsv, annotationData):
    ''' Extracts a feature vector from a datum and the data '''

    def sectionType(section):
        ''' Returns a string identifying the section type '''
        section = section.lower()

        if section.startswith('abstract'):
            return 'abstract'
        elif section.startswith('s'):
            return 'section'
        elif section.startswith('fig'):
            return 'figure'
        elif section == "":
            return 'section'
        else:
            return section

    # Distance in sections
    sections = annotationData['sections']
    titles = annotationData['titles']

    secSlice = sections[datum.evtIx:datum.ctxIx+1]
    changes = 0
    if len(secSlice) > 0:
        currSec = secSlice[0]
        for s in secSlice[1:]:
            if currSec != s:
                changes += 1
                currSec = s
    # Location relative
    ret = {
        'distance':float(abs(datum.evtIx - datum.ctxIx))/len(tsv),
        'sameSection':changes == 0,
        'ctxFirst':(datum.evtIx < datum.ctxIx),
        'sameLine':(datum.evtIx == datum.ctxIx),
        # 'ctxType':datum.ctx[0].upper(),
        # 'ctxSecitonType':sectionType(sections[datum.ctxIx]),
        # 'evtSecitonType':sectionType(sections[datum.ctxIx]),
        # 'ctxInTitle':titles[datum.ctxIx],
        'evtHasCitation':titles[datum.evtIx],
        'ctxHasCitation':titles[datum.ctxIx],
    }

    return ret

def machineLearning(X, y, clusters, X_test, y_test, clusters_test, fnames, crossvalidation):
    ''' Do the ML stuff
         - clusters is a variable that is used to implement type 2 eval:
           We only consider the set of unique refs per sentence, instead of multiple equivalent refs as in type 1
           They key is the index in the design matrix and the value the index in the reduced vector'''

    # Normalize the data in-place
    normalize(X, norm='l2', copy=False)
    normalize(X_test, norm='l2', copy=False)

    # Use a logistic regression model. Has regularization with l2 norm, fits the intercept
    lr = LogisticRegression(penalty='l1', C=.1)
    lr_test = LogisticRegression(penalty='l1', C=.1)


    if not crossvalidation:
        lr_test.fit(X, y)
        predictions = np.zeros(len(set(clusters_test.values())))
        new_y = np.zeros(len(set(clusters_test.values())))

        predicted = lr_test.predict(X_test)
        coef = lr_test.coef_
        for p, t, j in zip(predicted, y_test, xrange(X_test.shape[0])):
            predictions[clusters_test[j]] = p
            new_y[clusters_test[j]] = t
        # TODO refactor this to avoid hiding the original y array
        y = new_y
    else:
            predictions, y, coef = lkocv(lr, X, y, X.shape[0]*.01, clusters) # Hold 1% of the data out each fold


    if coef is not None:
        weigths = ["%s:%.3f std:%.3f" % (fn, c, s) for fn, c, s in zip(fnames, coef.mean(axis=0), coef.std(axis=0))]
        print
        print "Logistic regression coeficients:\n%s" % "\n".join(weigths)
        print

    labels = ["Isn't context", "Is context"]
    accuracy = metrics.accuracy_score(y, predictions)
    report = metrics.classification_report(y, predictions, target_names=labels)
    confusion = metrics.confusion_matrix(y, predictions)

    print report
    print "Accuracy: %.2f\n" % accuracy
    print "Confusion matrix:\n\t\tIsn't context\tIs context\nIsn't context\t    %i\t\t  %i\nIs context\t    %i\t\t  %i" % (confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1])

    # Print the classification ratios
    tp, fp, tn, fn = 0, 0, 0, 0
    for predicted, real in zip(predictions, y):
        if real == 1:
            if predicted == 1:
                tp += 1
            else:
                fn += 1
        else:
            if predicted == 1:
                fp += 1
            else:
                tn += 1

    positives = predictions.sum()
    negatives = y.shape[0] - positives

    print
    print "True positives ratio:\t%f" % (tp/positives)
    print "False positives ratio:\t%f" % (fp/positives)
    print "True negatives ratio:\t%f" % (tn/negatives)
    print "False negatives ratio:\t%f" % (fn/negatives)

    # Return a list of tuples (Index in the data list, error type) for debuging
    errors = []
    for ix, predicted, real in zip(xrange(len(y)), predictions, y):
        if predicted == 1 and real == 0:
            errors.append((ix, 'FP'))
        elif predicted == 0 and real == 1:
            errors.append((ix, 'FN'))
        elif predicted == 1 and real == 1:
            errors.append((ix, 'TP'))
        elif predicted == 0 and real == 0:
            errors.append((ix, 'TN'))

    return errors


def lkocv(predictor, X, y, k, clusters):
    ''' Does LKO-CV over the data and returns predictions '''

    predictions = np.zeros(len(set(clusters.values())))
    new_y = np.zeros(len(set(clusters.values())))

    # Cross validation
    cv = cross_validation.KFold(y.shape[0], n_folds=y.shape[0]//k, shuffle=True)
    coef = np.zeros((len(cv), X.shape[1]))

    for ix, split in enumerate(cv):
        train_ix, test_ix = split
        predictor.fit(X[train_ix], y[train_ix])
        predicted = predictor.predict(X[test_ix])
        for p, t, j in zip(predicted, y[test_ix], test_ix):
            predictions[clusters[j]] = p
            new_y[clusters[j]] = t
        # predictions[test_ix] = predicted
        coef[ix] = predictor.coef_[0]

    return predictions, new_y, coef

def parse_data(paths, annDir):
    ''' Reads and process stuff '''

    import ipdb; ipdb.set_trace()
    # Set this to false to generate negative examples instead of reach context annotations
    use_reach = True

    # Read everything
    accumulated_data = []
    vectors = []
    labels = []

    hashes = {}
    accumulator = 0
    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        tsv = parseTSV(path)
        data = extractData(tsv, path, use_reach)
        annotationData = extractAnnotationData(pmcid, annDir)

        if use_reach:
            # We need negative examples
            negatives = generateNegativesFromNER(data, annotationData)
            data = set(list(data) + negatives)

        for ix, datum in enumerate(data):
            accumulated_data.append(datum)
            vector = createFeatures(datum, data-{datum}, tsv, annotationData)
            vectors.append(vector)
            # Generate clusters for type 2 eval
            hashes[accumulator+ix] = hash(datum)

        labels += [datum.label for datum in data]
        accumulator += len(data)

    return labels, vectors, hashes, accumulated_data

def main(paths, annDir, testingIds, eval_type=EVAL2, crossvalidation=False):
    ''' Puts all together '''

    training_paths, testing_paths = [], []
    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        if pmcid in testingIds:
            testing_paths.append(path)
        else:
            training_paths.append(path)

    training_labels, training_vectors, training_hashes, training_data = parse_data(training_paths, annDir)
    testing_labels, testing_vectors, testing_hashes, testing_data = parse_data(testing_paths, annDir)

    print "General classifier"
    errors = pipeline(training_labels, training_vectors, training_hashes, testing_labels, testing_vectors, testing_hashes, eval_type, crossvalidation)

    return map(lambda e: (training_data[e[0]], e[1]), errors)

def pipeline(labels, vectors, hashes, testing_labels, testing_vectors, testing_hashes, eval_type, crossvalidation):

    dv = DictVectorizer()
    dv.fit(vectors + testing_vectors)

    X = dv.transform(vectors)
    y = np.asarray(labels)

    X_test = dv.transform(testing_vectors)
    y_test = np.asarray(testing_labels)

    if eval_type == EVAL1:
        clusters = {i:i for i in xrange(X.shape[0])}
        testing_clusters = {i:i for i in xrange(X_test.shape[0])}
    else:
        # Type two eval indexes
        indexes = {h:ix for ix, h in enumerate(set(hashes.values()))}
        clusters = {i:indexes[hashes[i]] for i in xrange(X.shape[0])}

        indexes_t = {h:ix for ix, h in enumerate(set(testing_hashes.values()))}
        testing_clusters = {i:indexes_t[testing_hashes[i]] for i in xrange(X_test.shape[0])}


    fnames = dv.feature_names_


    print "Total positive instances: %i\tTotal negative instances: %i" % (y[y == 1].shape[0], y[y == 0].shape[0])
    # Train and test a classifier
    errors = machineLearning(X, y, clusters, X_test, y_test, testing_clusters, fnames, crossvalidation)
    print

    return errors

    # vSpecies = [dict(v) for v in vectors if v['ctxType'] == 'S']
    # lSpecies = [label for label, v in zip(labels, vectors) if v['ctxType'] == 'S']
    #
    # for v in vSpecies:
    #     del v['ctxType']
    #
    # dv = DictVectorizer()
    #
    # X = dv.fit_transform(vSpecies)
    # y = np.asarray(lSpecies)
    #
    # fnames = dv.feature_names_

    # print "Species classifier"
    # print "Total positive instances: %i\tTotal negative instances: %i" % (y[y == 1].shape[0], y[y == 0].shape[0])
    # # Train and test a classifier
    # machineLearning(X, y, fnames)
    # print
    #
    # vCells = [dict(v) for v in vectors if v['ctxType'] == 'C']
    # lCells = [label for label, v in zip(labels, vectors) if v['ctxType'] == 'C']
    #
    # for v in vCells:
    #     del v['ctxType']
    #
    # dv = DictVectorizer()
    #
    # X = dv.fit_transform(vCells)
    # y = np.asarray(lCells)
    #
    # fnames = dv.feature_names_
    #
    # print "Cell classifier"
    # print "Total positive instances: %i\tTotal negative instances: %i" % (y[y == 1].shape[0], y[y == 0].shape[0])
    # # Train and test a classifier
    # machineLearning(X, y, fnames)
    # print
    #
    # vTissue = [dict(v) for v in vectors if v['ctxType'] == 'T']
    # lTissue = [label for label, v in zip(labels, vectors) if v['ctxType'] == 'T']
    #
    # for v in vTissue:
    #     del v['ctxType']
    #
    # dv = DictVectorizer()
    #
    # X = dv.fit_transform(vTissue)
    # y = np.asarray(lTissue)
    #
    # fnames = dv.feature_names_
    #
    # print "Tissue classifier"
    # print "Total positive instances: %i\tTotal negative instances: %i" % (y[y == 1].shape[0], y[y == 0].shape[0])
    # # Train and test a classifier
    # machineLearning(X, y, fnames)
    # print


# Entry point
if __name__ == "__main__":
    directory = sys.argv[1]
    annDir = sys.argv[2]
    testing_ids = {s[:-1] for s in open(sys.argv[3])} if len(sys.argv) > 3 else set()
    paths = glob.glob(os.path.join(directory, '*.tsv'))
    #paths = ['/Users/enoriega/Dropbox/Context Annotations/curated tsv/PMC2063868_E.tsv']

    errors = main(paths, annDir, testing_ids)
