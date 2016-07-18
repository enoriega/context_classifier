''' This is the file that implements the vector aggregation of the features for each event mention and context type
    It does cross validation over all the documents too '''


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
from model import *
from scipy.stats import bernoulli, ttest_1samp
from unbalanced_dataset import *
from sklearn.metrics import f1_score
from sklearn.feature_selection import RFE
from scipy.sparse import vstack, hstack
from classification_results import *
import classifier
import os
from model import *

def in_neighborhood(datum, intervals):
    ''' Used to filter a datum if it's not within the neighborhood of (min, max) '''

    for interval in intervals:
        minIx, maxIx = interval
        if datum.ctxIx >= minIx and datum.ctxIx <= maxIx:
            return True

    return False

def parse_data(paths, annDir, use_reach, relabeling, skip_amount=10):
    ''' Gets the Datum objects *before* extracting features '''

    # Read everything
    accumulated_data = {}

    #accumulator = 0
    total_pos, total_neg = 0, 0
    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        tsv = parseTSV(path)
        annotationData = extractAnnotationData(pmcid, annDir)
        data = extractData(tsv, path, annotationData, use_reach)

        # Attach the tsv and the annotation data to each datum
        for datum in data:
            datum.tsv = tsv
            datum.annotationData = annotationData

        # Skip this paper if it has less than 10 positive annotations
        if len([d for d in data if d.label == 1]) < skip_amount:
            continue


        augmented = data

        # We need negative examples
        negatives = generateNegativesFromNER(data, annotationData, relabeling)

        if len(negatives) > 0:

            size = len(negatives)
            negatives = list(np.random.choice(negatives, size=size if size <= len(negatives) else len(negatives), replace=False))

            # Attach the tsv and the annotation data to each datum
            for datum in negatives:
                datum.tsv = tsv
                datum.annotationData = annotationData

            augmented += negatives

        pos = sum([1 for a in augmented if a.label == 1])
        neg = sum([1 for a in augmented if a.label == 0])

        total_pos += pos
        total_neg += neg

        print '%s Positives: %i Negatives: %i Total: %i' % (pmcid, pos, neg, pos+neg)
        accumulated_data[pmcid] = augmented

    print
    print 'All papers. Positives: %i Negatives: %i Total: %i' % (total_pos, total_neg, total_pos+total_neg)
    print
    return accumulated_data


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
    ret = classifier.feda(ctxType, features)
    return ret

def cluster_dependencies(deps):
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
    if tag.startswith('NN'):
        return 'NN'
    elif tag.startswith('VB'):
        return 'VB'
    elif tag in {',', '-RRB-'}:
        return 'BOGUS'
    else:
        return tag

# Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

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


def discourse_path(datum, annotationData):
    def contains(i, interval):
        start, end = interval
        if i >= start and i <= end:
            return True
        else:
            return False

    def edu_contains(sen, tok, edu):
        # TODO: Continue here!!
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
        # Offset is necessary because the tree sentence ix starts in 0

        ctxSen = datum.ctxIx - offset
        evtSen = datum.evtIx - offset

        ctxToken, evtToken = datum.ctxToken, datum.evtToken

        ctxPath = find_leaf(ctxSen, ctxToken, disc)
        evtPath = find_leaf(evtSen, evtToken, disc)

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



    alldisc = annotationData['disc']
    for k in alldisc.keys():
        if contains(datum.ctxIx, k):
            ctxKey = k
        if contains(datum.evtIx, k):
            evtKey = k

    if ctxKey == evtKey:
        disc = alldisc[ctxKey]
        offset = ctxKey[0]
        path = common_ancestor(datum, offset, disc)
        return path
    else:
        return []

def create_features(data):
    ''' Creates a feature vector and attaches it to the datum object '''

    for datum in data:
        datum.features = generate_datum_features(datum, datum.tsv, datum.annotationData)
        datum.doc_features = generate_doc_features(datum, datum.tsv, datum.annotationData)

def generate_doc_features(datum, tsv, annotationData):

    # Mention counts
    cid = datum.ctxGrounded
    mention_counts = datum.annotationData['ctxCounts']

    ctx_freq = mention_counts[cid]

    features = {
        'ctx_freq':ctx_freq
    }

    return features

def vectorize_data(data):
    ''' Creates Numpy feature vectors out of datum objects '''

    cv, cv2 = DictVectorizer(), DictVectorizer()
    X, X2 = cv.fit_transform([d.features for d in data]), cv2.fit_transform([d.doc_features for d in data])
    # Normalize the matrix
    normalize(X, norm='max', copy=False)
    normalize(X2, norm='max', copy=False)

    for i, datum in enumerate(data):
        datum.vector = hstack([X[i, :], X2[i, :]])
        # datum.vector = hstack([X[i, :]])

    return cv, cv2

def crossval_baseline(folds):
    # This is essentially a "one-hit-all" evaluation of policy 4

    results = {}

    for fold_name, data in folds:
        predictions = {}
        truths = {}

        for datum in data:
            prediction = classifier.policy(datum, 6)
            truth = datum.label
            key = (datum.evt, datum.ctxGrounded)

            if key in predictions:
                # If the prediction if false, but the value is true,
                # lets override it. Otherwise the result is the same
                if predictions[key] == 0:
                    predictions[key] = prediction

                if truths[key] == 0:
                    truths[key] = truth
            else:
                # Store a result anyway
                predictions[key] = prediction
                truths[key] = truth

        keys = predictions.keys()
        result = ClassificationResults(fold_name, [truths[k] for k in keys], [predictions[k] for k in keys], [k for k in keys])
        results[fold_name] = result

    return results


def add_vectors(data, average=False):
    ''' Adds all the vectors from an event mention to a context type
        Optionally averages them '''

    def key(datum): return datum.namespace, datum.evt, datum.ctxGrounded

    data = sorted(data, key=key)
    groups = it.groupby(data, key)

    new_points, vectors, labels = [], [], []

    # Group them by event mention and ctx type
    for k, v in groups:
        new_points.append(k)
        local_vectors, local_labels = [], []
        for datum in v:
            local_vectors.append(datum.vector)
            local_labels.append(datum.label)

        # Add them together
        vector = sum(local_vectors)
        # Average them if specified
        if average:
            vector /= len(local_vectors)

        vectors.append(vector)

        # Get the label
        label = 1 if sum(local_labels) >= 1 else 0
        labels.append(label)

    return new_points, vstack(vectors), np.asarray(labels)


def train_eval_model(name, X_train, X_test, y_train, y_test, point_labels):
    ''' Configure the selected algorithm here and return a ClassificationResults object and a feature mask '''

    verbose = False
    # Edit the algorithm here
    # model = Perceptron(penalty='l2')
    algorithm = LogisticRegression(penalty='l2', C=.1)
    # model = SVC(verbose=verbose, kernel='linear', C=1)
    # algorithm = SVC(verbose=verbose, kernel='rbf', C=1)
    # model = SVC(verbose=verbose, kernel='poly', degree=3, C=50)
    # model = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=2)
    # model = RandomForestClassifier(n_estimators=10, max_depth=2)
    #########################

    algorithm.fit(X_train, y_train)
    predictions = algorithm.predict(X_test)
    coefficients = algorithm.coef_

    # Recursive feature selection
    # selector = RFE(algorithm, step=25, verbose=1)
    # selector = selector.fit(X_train, y_train)
    # print selector.support_.astype(int).sum()
    # Dimension-reduced data
    # mask = selector.support_
    # X_train, X_test = X_train[:, mask], X_test[:, mask]
    # Ensemble
    # ensemble = BaggingClassifier(LogisticRegression(penalty='l2', C=5), n_estimators=50, max_samples=1.0, verbose=2, n_jobs=3)
    # ensemble.fit(X_train, y_train)
    # coefficients = np.asarray([e.coef_[0] for e in ensemble.estimators_]).mean()
    # predictions = ensemble.predict(X_test)

    # Ensemble
    # ensemble = BaggingClassifier(LogisticRegression(penalty='l2', C=5), n_estimators=50, max_samples=.70, verbose=2, n_jobs=-1)
    # ensemble.fit(X_train, y_train)
    # coefficients = np.asarray([e.coef_[0] for e in ensemble.estimators_]).mean()
    # predictions = ensemble.predict(X_test)

    # coefficients = selector.estimator_.coef_[0]
    # predictions = selector.predict(X_test)


    mask = np.ones(X_train.shape[1]).astype(bool)
    return ClassificationResults(name, y_test, predictions, point_labels), mask, coefficients


def crossval_model(folds, limit_training, balance_dataset):

    aggregated_data = {}
    for fold_name, data in folds.iteritems():
        aggregated_data[fold_name] = add_vectors(data, False)


    results, masks, coeffictients = {}, {}, {}

    tests = []

    for fold_name in folds:
        point_labels, X_test, y_test = aggregated_data[fold_name]

        tests.append(X_test)


        X_train = vstack([aggregated_data[f][1] for f in aggregated_data if f != fold_name])
        y_train = np.concatenate([aggregated_data[f][2] for f in aggregated_data if f != fold_name])

        if balance_dataset:
            # Downsample negatives
            k = 3
            neg_ix = np.where(y_train == 0)[0]
            pos_ix = np.where(y_train == 1)[0]
            subsampled_negs = np.random.choice(neg_ix, size=int(len(pos_ix)*k), replace=False)
            new_ix = np.concatenate([subsampled_negs, pos_ix])
            # np.random.shuffle(new_ix)
            X_train = X_train[new_ix, :]
            y_train = y_train[new_ix]

            # # Upsample positives
            # l = 1.5
            # pos_ix = np.where(y_train == 1)[0]
            # neg_ix = np.where(y_train == 0)[0]
            # upsampled_pos = np.random.choice(pos_ix, size=int(len(pos_ix)*l), replace=True)
            # new_ix = np.concatenate([upsampled_pos, neg_ix])
            # np.random.shuffle(new_ix)
            # X_train = X_train[new_ix, :]
            # y_train = y_train[new_ix]


        print "Testing on %s ..." % fold_name
        print "%i positive - %i negative" % (y_train.sum(), len(y_train)-y_train.sum())
        results[fold_name], masks[fold_name], coeffictients[fold_name] = train_eval_model(fold_name, X_train, X_test, y_train, y_test, point_labels)


    return results, masks, coeffictients, vstack(tests)


# Entry point
if __name__ == "__main__":

    # Curated TSV directory
    directory = sys.argv[1]
    # Reach's annotation directory
    annDir = sys.argv[2]

    paths = glob.glob(os.path.join(directory, '*.tsv'))

    # CONFIG ###
    use_reach = True # Use reach's context extractions to extend the data set
    relabeling = parsing.RELABEL # Relabel alternative examples if they have the same type as one of Xia's choices for it's event
    limit_training = False # Only use the training examples that are in the neighborhood of the golden annotations
    # This is been deprecated, as we use aggregation now
    # one_hit_all = True # If one datum is classified positively, all data with the same context grounded id are postclassified as positive
    balance_dataset = True # Use a 1:K ratio of positive to negative examples during training
    ############

    print "Parsing data"
    cv_folds = parse_data(paths, annDir, use_reach = use_reach, relabeling=relabeling)

    print "Extracting features"
    create_features(it.chain(*cv_folds.values()))

    # print "Baseline cross validation"
    policy_results = crossval_baseline(cv_folds.iteritems())

    print "Vectorizing features"
    vectorizer, vectorizer2 = vectorize_data(list(it.chain(*cv_folds.values())))

    print "# of features: %i" % cv_folds.values()[0][0].vector.shape[1]
    print "Machine Learning cross validation"
    model_results, masks, coefficients, X  = crossval_model(cv_folds,\
                            limit_training=limit_training, balance_dataset=balance_dataset)

    macro_model, micro_model = MacroAverage("Macro model", model_results.values()), MicroAverage("Micro model", model_results.values())
    macro_policy, micro_policy = MacroAverage("Macro policy", policy_results.values()), MicroAverage("Micro policy", policy_results.values())
    #
    os.system('say "your program has finished"')
