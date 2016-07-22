''' Fuctions to parse the files and generate data structures '''

from __future__ import division
import csv, glob, sys, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
import operator
import networkx as nx
from random import shuffle
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn import cross_validation, metrics
from sklearn.feature_extraction import DictVectorizer
from context import triggers

EVAL1="ref"
EVAL2="sentence"

NO=0
RELABEL=1
EXCLUDE=2
DEBUG=False



def get_pmcid(name):
    ''' returns the pmcid out of a tsv file name '''
    pmcid = name.split(os.path.sep)[-1].split('.')[0]

    return pmcid

class Datum(object):
    ''' Represents an event/candidate context mention pair '''


    def __init__(self, namespace, evtIx, ctxIx, ctx, ctxGrounded, ctxToken, evt, evtToken, label, golden):
        self.namespace = namespace
        self.evtIx = evtIx
        self.ctxIx = ctxIx
        self.ctx = ctx
        self.ctxToken = ctxToken
        self.label = label
        self.evt = evt
        self.evtToken = evtToken
        self.ctxGrounded = ctxGrounded
        self.golden = golden
        self._hash = None
        self.tsv = None
        self.annotationData = None
        self.vector = None

    def __eq__(self, other):
        if isinstance(other, Datum):
            if self.namespace == other.namespace and \
                self.evtIx == other.evtIx and \
                self.ctxIx == other.ctxIx and \
                self.evt == other.evt and \
                self.ctx == other.ctx and \
                self.evtToken == other.evtToken and \
                self.ctxToken == other.ctxToken and \
                self.ctxGrounded == other.ctxGrounded:
                return True
            else:
                return False
        else:
            return False

    def __neq__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.namespace) + \
                hash('EvtIx %i' % self.evtIx) + \
                hash('CtxIx %i' % self.ctxIx) + \
                hash(self.ctx) + \
                hash(self.evt) + \
                hash('EvtToken %i' % self.ctxToken) + \
                hash('CtxToken %i' % self.evtToken) + \
                hash(self.ctxGrounded)

        return self._hash

    def __index__(self):
        return self.__hash__()

    def __str__(self):
        return "%s line %i %i %s %s %i %i %s %i %i" % (self.namespace, self.evtIx, self.evtToken, self.evt, self.ctx, self.ctxIx, self.ctxToken, self.ctxGrounded, self.label, self.golden)

    def __repr__(self):
        return str(self)


def extractData(tsv, name, annotationData, true_only=False):
    ''' Reads a parsed TSV and yields a sequence of data
     with positive and negative examples'''

    def isEvent(s):
        return len(s) > 0 and s[0].upper().startswith('E')

    def isContext(s):
        return len(s) > 0 and not isEvent(s)

    sections = annotationData['real_sections']
    manual_ctx = annotationData['manual_context_intervals']
    event_triggers = annotationData['manual_event_triggers']
    sentences = annotationData['sentences']

    events, context = [], []
    line_counter = 0
    for i, x in enumerate(tsv):
        # Filter out figures
        if sections[i].startswith('fig'):
            continue

        tbs, ix, cxs, cxgrounding = x['tbAnn'], line_counter, x['ctx'], x['ctxId']
        contextCounter = 0
        for tb in tbs:
            if isEvent(tb):
                for cx in cxs:
                    events.append((tb, ix, cx))
            elif isContext(tb):
                # This would be context annotations
                try:
                    context.append((tb, ix, cxgrounding[contextCounter]))
                    contextCounter += 1
                except:
                    print "Error in grounding id for file %s in line %i" % (name, i)

        line_counter += 1

    eLines = {e[0]:e[1] for e in events}
    cLines = {c[0]:(c[1], c[2]) for c in context}

    # Build the manual event token indices dictionary
    manual_evt = {}
    for line_num, items in it.groupby(eLines.iteritems(), lambda x: x[1]):
        sentence = sentences[line_num]
        trigger_nums = find_evt_anchors(sentence, triggers)

        event_ids = [i[0] for i in items]

        if len(trigger_nums) < len(event_ids):
            if DEBUG:
                print 'DEBUG: %s Line %i has fewer triggers than events' % (name ,line_num)

        for eid, tn in zip(event_ids, trigger_nums):
            manual_evt[eid] = tn

    # Generate negative examples
    sortedEvents = sorted(events, key=lambda x: x[0])
    groups = {k:list({x[2] for x in g}) for k, g in itertools.groupby(sortedEvents, key=lambda x: x[0])}

    sortedContext = [c[0] for c in sorted(context, key=lambda x: x[1])]

    def getAllOtherContext(location, excluded):
        ''' Pick all the other contexts from Xia's annotations '''

        candidateContexts = [(k, abs(v[0]-location)) for k, v in cLines.iteritems() if k not in excluded]
        if len(candidateContexts) > 0:

            return {c[0] for c in candidateContexts}
        else:
            return None


    # Extract data points
    true, false = [], []


    added = set()
    missing_manual_ctx = set()
    for e in events:
        evt, line, ctx = e
        if (evt, ctx) not in added:
            # Get the context line
            try:
                cLine, cGrounding = cLines[ctx]

                try:
                    ctx_token = manual_ctx[ctx]
                    try:
                        evt_token = manual_evt[evt]
                        true.append(Datum(name, line, cLine, ctx, cGrounding, ctx_token, evt, evt_token, 1, golden=True))
                    except:
                        if DEBUG:
                            missing_manual_ctx.add("DEBUG: Manual anchor evt missing %s %s" % (evt, name))
                        # missing_manual_ctx.add(sentences[line])
                except:
                    # pass
                    if DEBUG:
                        missing_manual_ctx.add("DEBUG: Manual anchor ctx missing %s %s" % (ctx, name))

            except:
                print "Key error %s %s" % (ctx, name)


            #print '%s: %s' % (k, [x[2] for x in g])

            localContext = groups[evt]

            added.add((evt, ctx))

            if not true_only:
                # Pick a negative example
                # ctx2s = getOtherContext(line, localContext)
                ctx2s = getAllOtherContext(line, localContext)

                if ctx2s is not None:
                    for ctx2 in ctx2s:
                        try:
                            cLine2, cGrounding2 = cLines[ctx2]
                            true.append(Datum(name, line, cLine2, ctx2, cGrounding2, manual_ctx[ctx2], evt, manual_evt[evt], 0, golden=False))
                        except e:
                            print e

    for s in missing_manual_ctx: print s

    return true+false

def generateNegativesFromNER(positives, annotationData, relabeling):
    ''' Generates all the negative examples out of the annotation data '''
    mentions = annotationData['mentions']

    # Generate a context label for contex
    alternatives = {}
    offset = 9000


    #TODO: Check this routine!! the CTX IX is WRONG!!
    for k, v in enumerate(mentions):
        for i in v:
            start, end, cid = i
            # Figure out the context type
            if cid.startswith('uaz:UA-ORG'):
                ctype = 'T'
            elif cid.startswith('uaz:UA-CLine' ):
                ctype = 'C'
            elif cid.startswith('taxonomy:'):
                ctype = 'S'
            elif cid.startswith('uaz:UA-CT'):
                ctype = 'C'
            elif cid.startswith('tissuelist:TS-'):
                ctype = 'C'
            else:
                print cid
                ctype = 'X'


            if ctype == 'X':
                continue

            alternatives['%s%i' % (ctype, offset)] = (k, cid, start)
            offset += 1



    negatives = []
    for datum in positives:
        for alternative, val in alternatives.iteritems():
            ix, cid, start = val
            # if datum.ctxIx != ix and datum.ctxGrounded.upper() != cid.upper():
            if datum.ctxIx != ix and datum.ctxToken != start:
                # Do the relabeling
                if relabeling != NO:
                    label = 1 if cid.upper() == datum.ctxGrounded else 0
                else:
                    label = 0


                if label == EXCLUDE:
                    continue

                new_datum = Datum(datum.namespace, datum.evtIx, ix, alternative.upper(), cid.upper(), int(start), datum.evt, datum.evtToken, label, golden=False)
                negatives.append(new_datum)



    return list(set(negatives))

not_permited_context = {'go', 'uniprot'}
not_permited_words = {'mum', 'hand', 'gatekeeper', 'muscle', 'spine', 'breast', 'head', 'neck', 'arm', 'leg'}

def map_2_filtered_ix(i, real_sections):
    ''' gets the correct index of the line after filtering out the figures '''

    sec_slice = real_sections[:i+1]

    figs = filter(lambda s: s.startswith('fig'), sec_slice)

    return i - len(figs)

def extractAnnotationData(pmcid, annDir):
    ''' Extracts data from annotations into a dictionary '''

    def parseBoolean(s):
        return True if s == 'true' else False

    pdir = os.path.join(annDir, pmcid)

    fsections = os.path.join(pdir, 'sections.txt')
    with open(fsections) as f:
        real_sections = [l[:-1] for l in f]
        sections = filter(lambda s: not s.startswith('fig'), real_sections)

    ftitles = os.path.join(pdir, 'titles.txt')
    with open(ftitles) as f:
        titles = map(lambda x: x[1], filter(lambda x: not x[0].startswith('fig'), zip(real_sections, [parseBoolean(l[:-1]) for l in f])))

    fcitations = os.path.join(pdir, 'citations.txt')
    with open(fcitations) as f:
        citations = map(lambda x: x[1], filter(lambda x: not x[0].startswith('fig'), zip(real_sections, [parseBoolean(l[:-1]) for l in f])))

    fdocnums = os.path.join(pdir, 'docnums.txt')
    with open(fdocnums) as f:
        docnums = map(lambda x: x[1], filter(lambda x: not x[0].startswith('fig'), zip(real_sections, [int(l[:-1]) for l in f])))

    fpostags = os.path.join(pdir, 'pos.txt')
    with open(fpostags) as f:
        postags = {int(k):v.split(' ') for k,v in [l[:-1].split('\t') for l in f]}

    fdeps = os.path.join(pdir, 'deps.txt')
    with open(fdeps) as f:
        # The file has an edge_list format to be parsed by networkx
        lines = [l[:-1].split('\t') for l in f]
        # Group the entries by their sentence index
        gb = it.groupby(lines, operator.itemgetter(0))
        # Create a networkx graph from the edge list for each sentence
        deps = {int(k):nx.parse_edgelist([x[1] for x in v], nodetype=int) for k, v in gb}

    fdiscourse = os.path.join(pdir, 'disc.txt')
    with open(fdiscourse) as f:
        # Parse the tsv. Cols: 1-starting sentence, 2-finishin sentence + 1, 3-Tree-like dict
        lines = [l[:-1].split('\t') for l in f]
        disc = dict()
        for s, e, t in lines:
            s, e = int(s), int(e)-1 # The second number is deliberately +1
            try:
                t = eval(t)
                disc[(s, e)] = t
            except:
                print "Error parsing discourse in %s for ix: %i-%i" % (fdiscourse, s, e)

    fmentions = os.path.join(pdir, 'mention_intervals.txt')

    with open(fmentions) as f:
        indices = defaultdict(list)
        for line in f:

            line = line[:-1]
            tokens = [t for t in line.split(' ') if t != '']
            ix = int(tokens[0])
            if ix < len(real_sections):
                if real_sections[ix].startswith('fig'):
                    continue

            ix = map_2_filtered_ix(ix, real_sections)

            intervals = []
            for t in tokens[1:]:
                x = t.split('%', 3)
                grounding_id = x[3].split(':')[0]

                word = x[2].lower()

                if grounding_id.lower() not in not_permited_context and word not in not_permited_words:
                    intervals.append((int(x[0]), int(x[1]), x[3]))


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

        tuples = []
        counter = 0
        for i, s in enumerate(real_sections):
            if s.startswith('fig'):
                counter += 1
            tuples.append((i, s, i-counter))

        #mentions = [indices[i] for i in xrange(max(indices.keys())+1)]
        mentions = [indices[j] for i, s, j in tuples if not s.startswith('fig')]

    # Do the mention counts
    # First count the reach mentions
    ctxCounts = Counter(n[2].upper() for n in it.chain(*[m for m in mentions if m]))
    # Normalize it
    total = sum(ctxCounts.values())
    for key in ctxCounts:
        ctxCounts[key] /= total


    fmanual_context_intervals = os.path.join(pdir, 'manual_context_intervals.txt')
    with open(fmanual_context_intervals) as f:
        manual_context_intervals = {}
        for l in f:
            l = l[:-1]
            line, interval, cid = l.split()
            interval = interval.split('-')
            manual_context_intervals[cid] = int(interval[0])


    # Do the manual event intervals
    fsentences = os.path.join(pdir, 'sentences.txt')
    with open(fsentences) as f:
        sentences = map(lambda x: x[1], filter(lambda x: not x[0].startswith('fig'), zip(real_sections, [l[:-1] for l in f])))
        manual_event_triggers = {i:find_evt_anchors(s, triggers) for i, s in enumerate(sentences)}


    return {
        'real_sections':real_sections,
        'sections':sections,
        'titles':titles,
        'citations':citations,
        'mentions':mentions,
        'docnums':docnums,
        'postags':postags,
        'deps':deps,
        'disc':disc,
        'manual_context_intervals':manual_context_intervals,
        'manual_event_triggers':manual_event_triggers,
        'sentences':sentences,
        'ctxCounts':ctxCounts
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


def find_evt_anchors(sentence, triggers):
    # Split by tokens
    tokens = sentence.strip().split()
    ret = []
    for i, t in enumerate(tokens):
        t = t.lower()
        for trigger in triggers:
            if trigger in t:
                ret.append(i)

    return ret
