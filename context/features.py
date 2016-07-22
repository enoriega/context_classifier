''' Code to generate features for classification '''

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
from context.features import *
import os
from context import *
from context.utils import *

def generate_datum_features(datum, tsv, annotationData):
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

    sections = annotationData['sections']
    titles = annotationData['titles']
    citations = annotationData['citations']
    docnums = annotationData['docnums']
    postags = annotationData['postags']
    deps = annotationData['deps']
    disc = annotationData['disc']
    sentences = annotationData['sentences']
    counts = annotationData['ctxCounts']

    # POS Tags
    ctxTag = cluster_postag(postags[datum.ctxIx][datum.ctxToken]) if datum.ctxIx in postags else None
    evtTag = cluster_postag(postags[datum.evtIx][datum.evtToken]) if datum.evtIx in postags else None

    # POS NGrams
    # if datum.ctxIx in postags:
    #     tags = postags[datum.ctxIx]
    #     x = []
    #     for i in range(datum.ctxIx-1, datum.ctxIx+2):
    #         if i >= 0 and i < len(tags):
    #             x.append(tags[i])
    #     ctxTag = '-'.join(x)
    # else:
    #     None
    #
    # if datum.evtIx in postags:
    #     tags = postags[datum.evtIx]
    #     x = []
    #     for i in range(datum.evtIx-1, datum.evtIx+2):
    #         if i >= 0 and i < len(tags):
    #             x.append(tags[i])
    #     evtTag = '-'.join(x)
    # else:
    #     None


    # Distance in sections
    secSlice = sections[datum.evtIx:datum.ctxIx+1]
    changes = 0
    if len(secSlice) > 0:
        currSec = secSlice[0]
        for s in secSlice[1:]:
            if currSec != s:
                changes += 1
                currSec = s

    # Distance in "paragraphs" (number of docs)
    distanceDocs = abs(docnums[datum.ctxIx] - docnums[datum.evtIx])

    # Context type, which is basically the KB where it was grounded from
    cid = datum.ctxGrounded
    if cid.startswith('TAXONOMY'):
        ctxType = 'Species'
    elif cid.startswith('TISSUELIST'):
        ctxType = 'Tissue'
    elif 'UA-CLINE' in cid:
        ctxType = 'CellLine'
    elif 'UA-CT' in cid:
        ctxType = 'CellType'
    elif 'UA-ORG' in cid:
        ctxType = 'Tissue'
    else:
        raise Exception("Undefined context type")

    # Dependecy path between ctx and evt
    dpath = dependency_path(datum, annotationData)
    dpath = cluster_dependencies(dpath)
    deps_len = len(dpath) if len(dpath) >= 1 else -1

    # POS Tag path
    ppath = postag_path(datum, annotationData)

    # Negation in the path between ctx and evt
    dep_negation = negation_in_dep_path(datum, annotationData)

    # Discourse path
    disc_path = discourse_path(datum, annotationData)
    disc_path = filter(lambda s: s not in {'Terminal'}, disc_path)
    disc_path = cluster_discourse(disc_path)
    disc_len = len(disc_path) if disc_path != [] else -1
    # Remove duplicates from the discourse path, to reduce dimensionality
    # disc_path = [k for k, v in it.groupby(disc_path)]
    # Remove "elaborates"
    # disc_path = filter(lambda s: s != 'elaboration', disc_path)


    ctx_doc_frequency = counts[cid] if cid in counts else 1

    #Binned distance in sentences
    if datum.evtIx == datum.ctxIx:
        distsents = "SAME"
    elif abs(datum.evtIx - datum.ctxIx) <= 13:
        distsents = "CLOSE"
    else:
        distsents = "FAR"

    #Binned distance in EDUs
    if disc_len == 0:
        binned_disclen = "SAME"
    elif abs(datum.evtIx - datum.ctxIx) <= 8:
        binned_disclen = "CLOSE"
    elif disc_len == -1:
        binned_disclen = "NULL"
    else:
        binned_disclen = "FAR"

    #Binned distance in dependencies
    if deps_len <= 5:
        binned_depslen = "CLOSE"
    elif disc_len == -1:
        binned_depslen = "NULL"
    else:
        binned_depslen = "FAR"

    #All binned distances
    all_bin_dists = '%s_%s_%s' % (distsents, binned_disclen, binned_depslen)

    # Location relative
    features = {
        # 'distance':'distsents:%i' % abs(datum.evtIx - datum.ctxIx),
        'binned_distance':distsents,
        # 'distanceDocs':'distdocs:%i' % distanceDocs,
        # 'sameSection':changes == 0,
        # 'binned_dist_section_change':'%s_%i' % (distsents, changes == 0),
        # 'evtFirst':(datum.evtIx < datum.ctxIx),
        # 'sameLine':(datum.evtIx == datum.ctxIx),
        # 'ctxType':ctxType
        # 'ctxSecitonType':sectionType(sections[datum.ctxIx]),
        # 'evtSecitonType':sectionType(sections[datum.ctxIx]),
        # 'ctxInTitle':titles[datum.ctxIx],
        # 'evtHasCitation':citations[datum.evtIx],
        # 'ctxHasCitation':citations[datum.ctxIx],
        # 'ctxInAbstract':'%s-%i' %  (distsents, sectionType(sections[datum.ctxIx]) == 'abstract'),
        # 'sameDocId':docnums[datum.ctxIx] == docnums[datum.evtIx],
        # 'ctxId':cid,
        'ctxTag':ctxTag,
        'evtTag':evtTag,
        # 'dependency_path':'|'.join(dpath),
        # 'dependency_length':'distdeps:%i' % deps_len,
        'binned_dependency_length':binned_depslen,
        # 'postag_path':'|'.join(ppath),
        # 'dep_negation':dep_negation,
        # 'discourse_path':'|'.join(disc_path),
        # 'disc_len':'distdisc:%i' % disc_len,
        'binned_disc_len':binned_disclen,
        # 'ctx_doc_frequency':ctx_doc_frequency,
        # 'all_binned_dists':all_bin_dists
    }

    # Dependency unigrams
    for dep in dpath:
        key = 'dep_path_unigrams:%s' % dep
        if key in features:
            features[key] = 1
        else:
            features[key] = 1

    # Dependency bigrams
    # for dep in pairwise(dpath):
    #     key = 'dep_path_bigrams:%s' % str(dep)
    #     if key in features:
    #         features[key] += 1
    #     else:
    #         features[key] = 1

    # POS unigrams
    for tag in ppath:
        key = 'pos_path_unigrams:%s' % tag
        if key in features:
            features[key] += 1
        else:
            features[key] = 1

    # Dependency contexts
    ctxContext = cluster_dependencies(dependency_context(datum, annotationData, ctx=True))
    # for x in ctxContext:
    #     key = 'ctxContext:%s' % x
    #     if key in features:
    #         features[key] += 1
    #     else:
    #         features[key] = 1
    #
    if 'neg' in '-'.join(ctxContext):
        features['ctxNegation'] = True

    evtContext = cluster_dependencies(dependency_context(datum, annotationData, ctx=False))
    # for x in evtContext:
    #     key = 'evtContext:%s' % x
    #     if key in features:
    #         features[key] += 1
    #     else:
    #         features[key] = 1
    #
    if 'neg' in '-'.join(evtContext):
        features['evtNegation'] = True

    # Discourse unigrams
    # if binned_disclen == "CLOSE":
    #     k = 1
    #     for i, d in enumerate(disc_path[:k] + disc_path[-k:]):
    #         key = 'discourse_unigrams:%i-%s' % (disc_len, d)
    #         if key in features:
    #             features[key] = 1
    #         else:
    #             features[key] = 1
    #
    #     # Discourse bigrams
    #     k = 3
    #     for i, d in enumerate(pairwise(disc_path[:k] + disc_path[-k:])):
    #         key = 'discourse_bigrams:%i-%s' % (disc_len, str(d))
    #         if key in features:
    #             features[key] = 1
    #         else:
    #             features[key] = 1


    # for i, d in enumerate(disc_path):
    #     key = 'discourse_path-%i:%s' % (i, str(d))
    #     if key in features:
    #         features[key] += 1
    #     else:
    #         features[key] = 1


    # ret = features
    ret = feda(ctxType, features)
    return ret

def dependency_path(datum, annotationData):
    ''' Extracts the shortest dependency path from an event to the context mention
        or returns None if it doesn't exist '''

    if datum.evtIx == datum.ctxIx:
        alldeps = annotationData['deps']
        deps = alldeps.get(datum.ctxIx, None)
        if deps is not None:
            try:
                path = nx.shortest_path(deps, datum.ctxToken, datum.evtToken)
                edges = pairwise(path)
                labels = map(lambda e: deps.get_edge_data(*e)['label'], edges)
                return labels
            except nx.NetworkXNoPath:
                return []
            except nx.NetworkXError as e:
                print "DEBUG: %s - NetworkX: %s" % (datum, e)
                return []
        else:
            print "DEBUG: Missing dependencies for %s" % datum
            return []
    else:
        return []

def postag_path(datum, annotationData):
    ''' Gets the POS tag path spannig from both mentions '''

    if datum.evtIx == datum.ctxIx:
        alldeps = annotationData['deps']
        deps = alldeps.get(datum.ctxIx, None)
        if deps is not None:
            try:
                path = nx.shortest_path(deps, datum.ctxToken, datum.evtToken)
                tags = annotationData['postags'][datum.ctxIx]
                labels = map(lambda e: tags[e], path)
                return labels
            except nx.NetworkXNoPath:
                return []
            except nx.NetworkXError as e:
                print "DEBUG: %s - NetworkX: %s" % (datum, e)
                return []
        else:
            print "DEBUG: Missing dependencies for %s" % datum
            return []
    else:
        return []

def get_neighborhood(deps, node):
    ''' Gets the "neigborhood" of dependencyes of a mention '''

    neighbors = deps.neighbors(node)
    ret = []
    for n in neighbors:
        neighbors2 = deps.neighbors(n)
        for n2 in neighbors2:
            ret.append((node, n, n2))
        # ret.append((node, n))

    return ret

def dependency_context(datum, annotationData, ctx=True):
    line_ix = datum.ctxIx if ctx else datum.evtIx
    token_ix = datum.ctxToken if ctx else datum.evtToken
    alldeps = annotationData['deps']
    deps = alldeps.get(line_ix, None)


    if deps is not None:
        try:
            paths = get_neighborhood(deps, token_ix)
            ret = []
            for p in paths:
                edges = pairwise(p)
                labels = map(lambda e: deps.get_edge_data(*e)['label'], edges)
                ret.append('-'.join(labels))
            return ret
        except nx.NetworkXNoPath:
            return []
        except nx.NetworkXError as e:
            print "DEBUG Dependency contexts: %s - NetworkX: %s" % (datum, e)
            return []
    else:
        return []

def negation_in_dep_path(datum, annotationData):
    ''' Logs for a "neg" dependency in the path between the mentions '''

    if datum.evtIx == datum.ctxIx:
        alldeps = annotationData['deps']
        deps = alldeps.get(datum.ctxIx, None)
        if deps is not None:
            try:
                path = nx.shortest_path(deps, datum.ctxToken, datum.evtToken)
                for s in path:
                    path = nx.shortest_path(deps, datum.ctxToken, datum.evtToken)
                    edges = pairwise(path)
                    labels = map(lambda e: deps.get_edge_data(*e)['label'], edges)
                    if 'neg' in labels:
                        print "Path negation!"
                        return True
                return False

            except nx.NetworkXNoPath:
                return False
            except nx.NetworkXError as e:
                print "DEBUG: %s - NetworkX: %s" % (datum, e)
                return False
        else:
            print "DEBUG: Missing dependencies for %s" % datum
            return False
    else:
        return False

def create_features(data):
    ''' Creates a feature vector and attaches it to the datum object
        Here we put together the mention participants' features and the
        document features'''

    for datum in data:
        datum.features = generate_datum_features(datum, datum.tsv, datum.annotationData)
        datum.doc_features = generate_doc_features(datum, datum.tsv, datum.annotationData)

def generate_doc_features(datum, tsv, annotationData):
    ''' Generate the features relative to to the document and not to the event mention - context mention pair '''

    # We only used context type frequency in a document for the paper

    # Mention counts
    cid = datum.ctxGrounded
    mention_counts = datum.annotationData['ctxCounts']

    ctx_freq = mention_counts[cid]

    features = {
        'ctx_freq':ctx_freq
    }

    return features
