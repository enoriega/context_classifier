''' Computes stats about the context annotations '''

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
        sentenceNumbers[pmcid] = len(tsv)
        data = extractData(tsv)
        annotationData = extractAnnotationData(pmcid, annDir)


        # Do stats
        fEvents, fContext = pandas(tsv, pmcid)

        eFrames.append(fEvents)
        cFrames.append(fContext)

    eFrame = pd.concat(eFrames)
    cFrame = pd.concat(cFrames)

    linesPerCtxCount = defaultdict(int)
    refs = {}
    proportionLinesWRef = {}
    proportionLinesWCtx = {}
    for name in eFrame.name.unique():
        # Number of refs
        refs[name] = eFrame[eFrame.name == name].shape[0]
        # proportion of sentences with context mentions
        numSentencesWithRef = eFrame[eFrame.name == name]["sentence"].unique().shape[0]
        proportionLinesWRef[name] =  numSentencesWithRef/sentenceNumbers[name]

        numSentencesWithCtx = cFrame[cFrame.name == name]["sentence"].unique().shape[0]
        proportionLinesWCtx[name] =  numSentencesWithCtx/sentenceNumbers[name]

        for s in eFrame[eFrame.name == name].sentence.unique():
            linesPerCtxCount[eFrame[(eFrame.name == name) & (eFrame.sentence == s)].shape[0]] += 1

    # Context per sentence: Cols: Num of contexts, rows: Number of sentences
    linesPerCtxCount = pd.Series(linesPerCtxCount)
    # Dist of context type
    # print cFrame[(cFrame.type == 's') | (cFrame.type == 'c') | (cFrame.type == 't')].type.value_counts()

    refs = pd.Series(refs)
    refs.sort(ascending=False)
    proportionLinesWRef = pd.Series(proportionLinesWRef)
    proportionLinesWRef.sort(ascending=False)
    proportionLinesWCtx = pd.Series(proportionLinesWCtx)
    proportionLinesWCtx.sort(ascending=False)
    linesPerCtxCount = pd.Series(linesPerCtxCount)
    linesPerCtxCount.sort(ascending=False)

    # Uncomment for stats
    # print refs
    # print proportionLinesWRef
    # print proportionLinesWCtx
    # print linesPerCtxCount

# Entry point
if __name__ == "__main__":
    directory = sys.argv[1]
    annDir = sys.argv[2]
    paths = glob.glob(os.path.join(directory, '*.tsv'))
    #paths = ['/Users/enoriega/Dropbox/Context Annotations/curated tsv/PMC2063868_E.tsv']

    main(paths, annDir)
