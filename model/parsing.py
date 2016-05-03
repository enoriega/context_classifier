''' Fuctions to parse the files and generate data structures '''

from __future__ import division
import csv, glob, sys, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
from random import shuffle
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn import cross_validation, metrics
from sklearn.feature_extraction import DictVectorizer

EVAL1="ref"
EVAL2="sentence"

def get_pmcid(name):
    ''' returns the pmcid out of a tsv file name '''
    pmcid = name.split(os.path.sep)[-1].split('.')[0]

    return pmcid

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

def extractData(tsv, name, true_only=False):
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

            if not true_only:
                # Pick a negative example
                ctx2s = getOtherContext(line, localContext)

                if ctx2s is not None:
                    for ctx2 in ctx2s:
                        try:
                            cLine2 = cLines[ctx2]
                            true.append(Datum(name, line, cLine2, ctx2, eid, 0))
                        except e:
                            print e

    return true+false

def generateNegativesFromNER(positives, annotationData):
    ''' Generates all the negative examples out of the annotation data '''
    mentions = annotationData['mentions']

    # Generate a context label for contex
    alternatives = {}
    offset = 9000
    for k, v in enumerate(mentions):
        for i in v:
            alternatives['S%i' % offset] = k
            offset += 1


    negatives = []
    for datum in positives:
        for alternative, ix in alternatives.iteritems():
            if datum.ctxIx != ix:
                new_datum = Datum(datum.namespace, datum.evtIx, ix, alternative, datum.eid, 0)
                negatives.append(new_datum)

    return negatives

not_permited_context = {'go', 'uniprot'}
not_permited_words = {'mum', 'hand', 'gatekeeper', 'muscle', 'spine', 'breast'}
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

    fmentions = os.path.join(pdir, 'mention_intervals.txt')

    with open(fmentions) as f:
        indices = defaultdict(list)
        for line in f:
            line = line[:-1]
            tokens = [t for t in line.split(' ') if t != '']
            ix = int(tokens[0])
            intervals = []
            for t in tokens[1:]:
                x = t.split('-')
                grounding_id = x[3].split(':')[0]
                #if len(x) >= 4: # Hack to handle absence of nsID (reach grounding bug)
                word = x[2].lower()
                if grounding_id not in not_permited_context and word not in not_permited_words:
                    intervals.append((int(x[0]), int(x[1])))

            # Merge succesive intervals
            merged = []
            if len(intervals) > 0:
                prev = intervals[0]
                for curr in intervals[1:]:
                    if prev[1] == curr[0]:
                        x = (prev[0], curr[1])
                        merged.append(x)
                        prev = x
                    else:
                        merged.append(prev)
                        merged.append(curr)
                        prev = curr

            # Pick only one interval per line
            # if len(merged) > 1:
            #     merged = [merged[0]]
            indices[ix] += merged



        mentions = [indices[i] for i in xrange(max(indices.keys())+1)]

    return {
        'sections':sections,
        'titles':titles,
        'citations':citations,
        'mentions':mentions
    }



def annotationStats(paths, annotationsDir):

    rows = []

    for p in paths:
        pid = get_pmcid(p)
        row = {}
        row['id'] = pid
        mentions = mentionCounts(os.path.join(annotationsDir, pid, 'mention_intervals.txt'))
        row['mentions'] = mentions.shape[0]
        row['species_mentions'] = mentions[mentions['type'] == 'Species'].shape[0]
        row['ct_mentions'] = mentions[mentions['type'] == 'CT'].shape[0]
        row['cl_mentions'] = mentions[mentions['type'] == 'CL'].shape[0]
        row['org_mentions'] = mentions[mentions['type'] == 'ORG'].shape[0]
        tsv = parseTSV(p)
        annotations = [d for d in extractData(tsv, p, False) if d.label == 1]
        row['refs'] = len(annotations)
        row['ctx_annotations'] = len({a.ctx for a in annotations})
        row['evt_annotations'] = len({a.eid for a in annotations})
        rows.append(row)

    return pd.DataFrame(rows).set_index('id')

def mentionCounts(path):
    def ctxType(s):
        if "taxonomy" in s:
            return "Species"
        elif "ua-ct" in s:
            return "CT"
        elif "org" in s:
            return "ORG"
        elif "ua-cl" in s:
            return "CL"

    mentions = []
    with open(path) as f:
        for line in f:
            line = line[:-1].lower()
            ix, intervals = line.split(' ', 1)
            intervals = [i for i in intervals.split(' ') if i != '']

            for interval in intervals:
                start, end, word, ctxid = interval.split('-', 3)
                t = ctxType(ctxid)
                if t is not None and word not in not_permited_words:
                    mentions.append({'type':t})

    ret = pd.DataFrame(mentions)
    return ret

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

def split_dataset(directory, training_size, num_samples=1):
    ''' Splits the dataset with proportion of 'training_size' "refs".
        training_size should be > 0 and < 1
        Returns two lists [[training_ids], [testing_ids]] '''

    # Complete dataset
    paths = glob.glob(os.path.join(directory, '*.tsv'))

    # All pmc ids
    ids = []

    numrefs = {}
    for path in paths:
        counter = 0
        pmcid = get_pmcid(path)
        ids.append(pmcid)
        tsv = parseTSV(path)
        data = extractData(tsv, path)
        # Count the number of positive examples
        positive = [d for d in data if d.label == 1]
        filtered = set()
        for p in positive:
            filtered.add(p.evtIx)

        numrefs[pmcid] = len(filtered)

    # Make ids a set to support set operations
    ids_set = set(ids)

    total_refs = sum(numrefs.values())

    # I'm going dumb with a brute force approach because the number or papers is small
    # Generate the powerset of all the pmcids
    ret = []

    for x in xrange(num_samples):
        shuffle(ids)
        sizes = [numrefs[pmcid] for pmcid in ids]
        for i in xrange(len(ids)):
            size = sum(sizes[:i])/total_refs
            if size >= training_size - .02 and size <= training_size + .02:
                candidate = set(ids[:i])
                ret.append((candidate, ids_set-candidate))
                print size
                break

    return ret
