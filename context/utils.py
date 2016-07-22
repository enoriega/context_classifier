''' Utilities for the experiment '''
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
from features import *
import os
from context import *

def policy(datum, k):
    ''' Deterministic policy for the baseline '''
    if abs(datum.ctxIx-datum.evtIx) <= k:
        return 1
    else:
        return 0

# See Daume III, Hal. "Frustratingly easy domain adaptation." arXiv preprint arXiv:0907.1815 (2009).
def feda(ctype, features):
    ''' Frustratingly easy domain adaptation '''

    new_dict = {}
    for k, v in features.iteritems():
        new_dict[k] = v
        new_dict['%s_%s' % (ctype, k)] = v

    return new_dict

def contains(i, interval):
    ''' Check whether i is in [a, b] '''

    start, end = interval
    if i >= start and i <= end:
        return True
    else:
        return False

def discourse_path(datum, annotationData):
    ''' Builds the path from the relation participants in a RST tree '''


    def edu_contains(sen, tok, edu):
        ''' Check to see if a pointer (sentence, token of sentence)
        is containes in a particular EDU of the tree '''

        start, end = edu['start'], edu['end']

        if start[0] == end[0]:
            interval = start[1], end[1]
            if sen == start[0]:
                return contains(tok, interval)
            else:
                return False
        else:
            if sen == start[0]:
                if tok >= start[1]:
                    return True
                else:
                    return False
            elif sen == end[0]:
                if tok <= end[1]:
                    return True
                else:
                    return False
            elif sen > start[0] and sen < end[0]:
                return True
            else:
                return False

    def find_leaf(sen, tok, edu):
        ''' Locates the leave of a mention, which is an EDU in the data structure '''
        if 'children' not in edu:
            if edu_contains(sen, tok, edu):
                return [(edu, "Terminal")]
            else:
                return None
        else:
            for c in edu['children']:
                l = find_leaf(sen, tok, c)
                if l is not None:
                    return [(edu, edu['label'])] + l
            return None
            raise Exception("DEBUG: Find leaf - Shouln't reach here")


    def common_ancestor(datum, offset, disc):
        ''' Finds the lowest common ancestor of both mentions in the tree
            and returns the path that connects them and goes through the ancestor '''

        # Offset is necessary because the tree sentence ix starts in 0

        ctxSen = datum.ctxIx - offset
        evtSen = datum.evtIx - offset

        ctxToken, evtToken = datum.ctxToken, datum.evtToken

        # Locate the mention leaves
        ctxPath = find_leaf(ctxSen, ctxToken, disc)
        evtPath = find_leaf(evtSen, evtToken, disc)

        # Build the path
        for c, e in zip(ctxPath, evtPath):
            if c == e:
                common = c
                ctxPath = ctxPath[1:]
                evtPath = evtPath[1:]

        ctxLabels = filter(lambda x: x != 'Terminal', map(lambda x: x[1], ctxPath))
        evtLabels = filter(lambda x: x != 'Terminal', map(lambda x: x[1], evtPath))

        # Remove the terminal labels and connect both via the common ancestor
        path = ctxLabels + [common[1]] + evtLabels

        return path

    # This is where the function actually begins

    # If both participant mentions aren't within the same discourse tree, finish early
    alldisc = annotationData['disc']
    for k in alldisc.keys():
        if contains(datum.ctxIx, k):
            ctxKey = k
        if contains(datum.evtIx, k):
            evtKey = k

    # Otherwise, find the common ancestor in the tree and enumerate the path
    if ctxKey == evtKey:
        disc = alldisc[ctxKey]
        # I hate this but is necessary
        offset = ctxKey[0]
        path = common_ancestor(datum, offset, disc)
        return path
    else:
        return []

def cluster_dependencies(deps):
    ''' Groups dependencies into functionally similar groups '''
    ret = []
    for d in deps:
        if d.startswith('prep'):
            ret.append('prep')
        elif d.startswith('conj'):
            ret.append('conj')
        elif d.endswith('obj'):
            ret.append('obj')
        elif d.endswith('mod'):
            ret.append('mod')
        elif 'subj' in d:
            ret.append('subj')
        else:
            ret.append(d)
    return ret

def cluster_discourse(disc):
    ''' Groups discourse relations into functionally similar groups '''

    ret = []
    for edu in disc:
        if edu in {'attribution', 'cause', 'enablement', 'manner-means'}:
            ret.append('causal')
        elif edu in {'topic-change', 'topic-comment'}:
            ret.append('topic')
        elif edu in {'joint', 'same-unit'}:
            ret.append('misc')
        else:
            ret.append(edu)

    return ret

def cluster_postag(tag):
    ''' Groups POS tags into functionally similar groups '''

    if tag.startswith('NN'):
        return 'NN'
    elif tag.startswith('VB'):
        return 'VB'
    elif tag in {',', '-RRB-'}:
        return 'BOGUS'
    else:
        return tag

# Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
# It builds bigrams
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)
