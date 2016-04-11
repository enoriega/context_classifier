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

def createFeatures(datum, otherData, tsv, annotationData):
    ''' Extracts a feature vector from a datum and the data '''

    # if datum.ctx[0].upper() == '4':
    #     print datum.ctx

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
        'ctxType':datum.ctx[0].upper(),
        'ctxSecitonType':sectionType(sections[datum.ctxIx]),
        'evtSecitonType':sectionType(sections[datum.ctxIx]),
        # 'ctxInTitle':titles[datum.ctxIx],
        # 'ctxOfOtherEvt':len({d for d in otherData if d.ctx == datum.ctx}) > 0,
        'evtHasCitation':titles[datum.evtIx],
        'ctxHasCitation':titles[datum.ctxIx],
    }

    return ret

def machineLearning(X, y, fnames):
    ''' Do the ML stuff '''

    # Normalize the data in-place
    normalize(X, norm='l2', copy=False)

    # Use a logistic regression model. Has regularization with l2 norm, fits the intercept
    lr = LogisticRegression(penalty='l2', C=.1)

    #predictions, y, coef = train_test(lr, X, y)
    predictions, y, coef = lkocv(lr, X, y, X.shape[0]*.01) # Hold 1% of the data out each fold

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


def lkocv(predictor, X, y, k):
    ''' Does LKO-CV over the data and returns predictions '''
    predictions = np.zeros(X.shape[0])

    # Cross validation
    cv = cross_validation.KFold(y.shape[0], n_folds=y.shape[0]//k, random_state=76654, shuffle=True)
    coef = np.zeros((len(cv), X.shape[1]))

    for ix, split in enumerate(cv):
        train_ix, test_ix = split
        predictor.fit(X[train_ix], y[train_ix])
        predictions[test_ix] = predictor.predict(X[test_ix])
        coef[ix] = predictor.coef_[0]

    return predictions, y, coef


def main(paths, annDir):
    ''' Puts all together '''

    # Read everything
    vectors = []
    labels = []

    eFrames, cFrames = [], []

    sentenceNumbers = {}

    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        tsv = parseTSV(path)
        data = extractData(tsv)
        annotationData = extractAnnotationData(pmcid, annDir)
        vectors += [createFeatures(datum, data-{datum}, tsv, annotationData) for datum in data]
        labels += [datum.label for datum in data]

    dv = DictVectorizer()

    X = dv.fit_transform(vectors)
    y = np.asarray(labels)



    fnames = dv.feature_names_

    print "General classifier"
    print "Total positive instances: %i\tTotal negative instances: %i" % (y[y == 1].shape[0], y[y == 0].shape[0])
    # Train and test a classifier
    machineLearning(X, y, fnames)
    print

    vSpecies = [dict(v) for v in vectors if v['ctxType'] == 'S']
    lSpecies = [label for label, v in zip(labels, vectors) if v['ctxType'] == 'S']

    for v in vSpecies:
        del v['ctxType']

    dv = DictVectorizer()

    X = dv.fit_transform(vSpecies)
    y = np.asarray(lSpecies)

    fnames = dv.feature_names_

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
    paths = glob.glob(os.path.join(directory, '*.tsv'))
    #paths = ['/Users/enoriega/Dropbox/Context Annotations/curated tsv/PMC2063868_E.tsv']

    main(paths, annDir)
