''' Fuctions to parse the files and generate data structures '''

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

EVAL1="ref"
EVAL2="sentence"

class Datum(object):
    ''' Represents an event/candidate context mention pair '''

    def __init__(self, namespace, evtIx, ctxIx, ctx, eid, label):
        self.namespace = namespace
        self.evtIx = evtIx
        self.ctxIx = ctxIx
        self.ctx = ctx
        self.label = label
        self.eid = eid

    def __hash__(self):
        s = self.namespace+str(self.evtIx)+str(self.ctxIx)+self.ctx
        return hash(s)

    def __str__(self):
        return "%s line %i %s %s" % (self.namespace, self.evtIx, self.eid, self.ctx)

    def __repr__(self):
        return str(self)

def saveErrors(errors, path):
    ''' Save the classification errors to a csv file '''
    labels = ['File', 'Line', 'EvtID', 'CtxID', 'EType']
    rows = []
    for err in errors:
        datum, etype = err
        rows.append([datum.namespace, datum.evtIx, datum.eid, datum.ctx, etype])

    with open(path, 'w') as f:
        writter = csv.writer(f)
        writter.writerow(labels)
        writter.writerows(rows)

def extractData(tsv, name):
    ''' Reads a parsed TSV and yields a sequence of data
     with positive and negative examples'''

    def isEvent(s):
        return len(s) > 0 and s[0].upper().startswith('E')

    def isContext(s):
        return len(s) > 0 and not isEvent(s)

    events, context = [], []
    for x in tsv:
        tbs, ix, cxs = x['tbAnn'], int(x['num'])-1, x['ctx']
        for tb in tbs:
            if isEvent(tb):
                for cx in cxs:
                    events.append((tb, ix, cx))
            elif isContext(tb):
                # This would be context annotations
                context.append((tb, ix))

    eLines = {e[0]:e[1] for e in events}
    cLines = {c[0]:c[1] for c in context}

    # Generate negative examples
    sortedEvents = sorted(events, key=lambda x: x[0])
    groups = {k:list({x[2] for x in g}) for k, g in itertools.groupby(sortedEvents, key=lambda x: x[0])}

    sortedContext = [c[0] for c in sorted(context, key=lambda x: x[1])]

    def getOtherContext(location, excluded, num=3):
        ''' Pick randomly another context with a probability proportional to it's distance from pivot '''

        candidateContexts = [(k, abs(v-location)) for k, v in cLines.iteritems() if k not in excluded]
        if len(candidateContexts) > 0:
            probs = np.asarray([x[1] for x in candidateContexts], dtype=float)
            probs /= probs.sum()

            choices = np.random.choice(len(candidateContexts), num if len(candidateContexts) > num else len(candidateContexts), p=probs, replace=False)

            # TODO: Really fix this!!
            x = filter(lambda a: not a.isdigit(), {candidateContexts[choice][0] for choice in choices})
            # for a in x:
            #     if a.isdigit():
            #         print x
            return x
        else:
            return None


    # Extract data points
    true, false = [], []
    added = set()
    for e in events:
        eid, line, ctx = e
        if (eid, ctx) not in added:
            # Get the context line
            try:
                cLine = cLines[ctx]
                true.append(Datum(name, line, cLine, ctx, eid, 1))
            except e:
                print e


            #print '%s: %s' % (k, [x[2] for x in g])

            localContext = groups[eid]

            added.add((eid, ctx))

            # Pick a negative example
            ctx2s = getOtherContext(line, localContext)

            if ctx2s is not None:
                for ctx2 in ctx2s:
                    try:
                        cLine2 = cLines[ctx2]
                        true.append(Datum(name, line, cLine2, ctx2, eid, 0))
                    except e:
                        print e

    return set(true+false)

def extractAnnotationData(pmcid, annDir):
    ''' Extracts data from annotations into a dictionary '''

    pdir = os.path.join(annDir, pmcid)

    fsections = os.path.join(pdir, 'sections.txt')
    with open(fsections) as f:
        sections = [l[:-1] for l in f]

    ftitles = os.path.join(pdir, 'titles.txt')
    with open(ftitles) as f:
        titles = [bool(l[:-1]) for l in f]

    fcitations = os.path.join(pdir, 'citations.txt')
    with open(fcitations) as f:
        citations = [bool(l[:-1]) for l in f]

    return {
        'sections':sections,
        'titles':titles,
        'citations':citations
    }

def parseTSV(path):
    ''' Parses a tsv file '''
    fieldNames = ['num', 'ctxId', 'ctxTxt', 'tbAnn', 'ctx', 'evtTxt', 'text']

    def split(seq):
        return [s.strip().upper() for s in seq.split(',')]

    rows = []
    with open(path) as f:
        dr = csv.DictReader(f, delimiter='\t', fieldnames=fieldNames)
        for ix, row in enumerate(dr):
            row['ctxId'] = split(row['ctxId']) if row['ctxId'] is not None else None
            row['ctxTxt'] = split(row['ctxTxt']) if row['ctxTxt'] is not None else None
            row['evtTxt'] = split(row['evtTxt']) if row['evtTxt'] is not None else None
            row['tbAnn'] = split(row['tbAnn']) if row['tbAnn'] is not None else None
            row['ctx'] = split(row['ctx']) if row['ctx'] is not None else None

            rows.append(row)

    return rows

def pandas(tsv, name):
    ''' Creates pandas data frames for rows '''

    def isEvent(s):
        return len(s) > 0 and s[0].upper().startswith('E')

    def isContext(s):
        return len(s) > 0 and not isEvent(s)

    events, context = [], []
    for x in tsv:
        tbs, ix, cxs = x['tbAnn'], int(x['num'])-1, x['ctx']
        for tb in tbs:
            if isEvent(tb):
                for cx in cxs:
                    events.append({"eid":tb, "name":name, "sentence":ix, "context":cx})
            elif isContext(tb):
                # This would be context annotations
                context.append({"cid":tb, "name":name, "sentence":ix, "type":tb[0].lower()})

    fEvents = pd.DataFrame(events)
    fContext = pd.DataFrame(context)
    # eLines = {e[0]:e[1] for e in events}
    # cLines = {c[0]:c[1] for c in context}

    return fEvents, fContext
