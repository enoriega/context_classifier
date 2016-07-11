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
from scipy.stats import bernoulli
from unbalanced_dataset import *
from sklearn.metrics import f1_score

np.random.seed(0)

def policy(datum, k):
    if abs(datum.ctxIx-datum.evtIx) <= k:
        return 1
    else:
        return 0

def createFeatures(datum, tsv, annotationData):
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

    # POS Tags
    ctxTag = postags[datum.ctxIx][datum.ctxToken] if datum.ctxIx in postags else None
    evtTag = postags[datum.evtIx][datum.evtToken] if datum.evtIx in postags else None

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


    # Location relative
    features = {
        'distance':'distsents:%i' % abs(datum.evtIx - datum.ctxIx),
        'distanceDocs':'distdocs:%i' % distanceDocs,
        'sameSection':changes == 0,
        'evtFirst':(datum.evtIx < datum.ctxIx),
        'sameLine':(datum.evtIx == datum.ctxIx),
        # 'ctxType':ctxType
        'ctxSecitonType':sectionType(sections[datum.ctxIx]),
        # 'evtSecitonType':sectionType(sections[datum.ctxIx]),
        'ctxInTitle':titles[datum.ctxIx],
        # 'evtHasCitation':citations[datum.evtIx],
        'ctxHasCitation':citations[datum.ctxIx],
        # 'ctxInAbstract':sectionType(sections[datum.ctxIx]) == 'abstract',
        'sameDocId':docnums[datum.ctxIx] == docnums[datum.evtIx],
        'ctxTag':ctxTag,
        'evtTag':evtTag,
    }

    ret = features
    # ret = feda(ctxType, features)
    return ret

# Frustratingly easy domain adaptation
def feda(ctype, features):
    new_dict = {}
    for k, v in features.iteritems():
        new_dict[k] = v
        new_dict['%s_%s' % (ctype, k)] = v

    return new_dict


def machineLearning(X, y, clusters, X_test, y_test, clusters_test, fnames, crossvalidation, training_data):
    ''' Do the ML stuff
         - clusters is a variable that is used to implement type 2 eval:
           We only consider the set of unique refs per sentence, instead of multiple equivalent refs as in type 1
           They key is the index in the design matrix and the value the index in the reduced vector'''

    # Normalize the data in-place
    normalize(X, norm='l2', copy=False)
    normalize(X_test, norm='l2', copy=False)

    # Use a logistic regression model. Has regularization with l2 norm, fits the intercept
    # lr = RandomForestClassifier(max_depth=10)
    # lr_test = RandomForestClassifier(max_depth=10)
    # lr = LogisticRegression(penalty='l2', C=10)
    # lr_test = LogisticRegression(penalty='l2', C=10)
    # lr = SVC(kernel='sigmoid')
    # lr_test = SVC(kernel='sigmoid')
    lr = Perceptron(penalty='l2')
    lr_test = Perceptron(penalty='l2')

    verbose = False
    ratio = float(np.count_nonzero(y==0)) / float(np.count_nonzero(y==1))

    smote = SMOTE(ratio=ratio, verbose=verbose, kind='regular')
    # smote = SMOTETomek(ratio=ratio, verbose=verbose)
    # smote = UnderSampler(verbose=verbose)
    # smote = TomekLinks(verbose=verbose)
    # smote = ClusterCentroids(verbose=verbose)
    # smote = NearMiss(version=1, verbose=verbose)
    # smote = NearMiss(version=2, verbose=verbose)
    # smote = NearMiss(version=3, verbose=verbose)
    # smote = NeighbourhoodCleaningRule(size_ngh=51, verbose=verbose)
    # smox, smoy = smote.fit_transform(X.todense(), y)
    smox, smoy = X, y

    print "Training class sizes: Is Context: %i\t Isn't Context: %s" % (smoy.sum(), smoy.shape[0]-smoy.sum())

    if not crossvalidation:
        lr_test.fit(smox, smoy)
        predictions = np.zeros(len(set(clusters_test.values())))
        new_y = np.zeros(len(set(clusters_test.values())))

        predicted = lr_test.predict(X_test)
        try:
            coef = lr_test.coef_
        except:
            coef = None

        for p, t, j in zip(predicted, y_test, xrange(X_test.shape[0])):
            predictions[clusters_test[j]] = p
            new_y[clusters_test[j]] = t
        # TODO refactor this to avoid hiding the original y array
        y = new_y
    else:
            predictions, y, coef = lkocv(lr, smox, smoy, smox.shape[0]*.01, clusters, training_data) # Hold 1% of the data out each fold
            # predictions, y, coef = lkocv(lr, smox, smoy, 10, clusters, training_data) # Hold 1% of the data out each fold


    # if coef is not None:
    #     weigths = ["%s:%.3f std:%.3f" % (fn, c, s) for fn, c, s in zip(fnames, coef.mean(axis=0), coef.std(axis=0))]
    #     print
    #     print "Logistic regression coeficients:\n%s" % "\n".join(weigths)
    #     print

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


error_hists, success_hists = [], []
f1_differences = []
def lkocv(predictor, X, y, k, clusters, data):
    ''' Does LKO-CV over the data and returns predictions '''

    predictions = np.zeros(len(set(clusters.values())))
    new_y = np.zeros(len(set(clusters.values())))

    # Cross validation
    cv = cross_validation.KFold(y.shape[0], n_folds=y.shape[0]//k, shuffle=True)
    coef = np.zeros((len(cv), X.shape[1]))


    included = set()
    for ix, split in enumerate(cv):
        train_ix, test_ix = split
        predictor.fit(X[train_ix], y[train_ix])
        predicted = predictor.predict(X[test_ix])

        error_hist, success_hist = [], []

        policy_predictions = []
        for i in test_ix:
            datum = data[i]
            policy_predictions.append(policy(datum, 3))

        for p, t, j in zip(predicted, y[test_ix], test_ix):
            predictions[clusters[j]] = p
            new_y[clusters[j]] = t
            datum = data[clusters[j]]

            # Do the histograms
            hbin = abs(datum.ctxIx - datum.evtIx)

            if datum.label == 1 and hash(datum) not in included:
                included.add(hash(datum))
                if p == t:
                    success_hist.append(hbin)
                else:
                    error_hist.append(hbin)

        # Record the F1 of the classifier and of the baseline
        predicted_f1 = f1_score(y[test_ix], predicted)
        baseline_f1 = f1_score(y[test_ix], np.asarray(policy_predictions))

        f1_differences.append(predicted_f1 - baseline_f1)

        error_hists.append(error_hist)
        success_hists.append(success_hist)
        # predictions[test_ix] = predicted
        coef[ix] = predictor.coef_[0]


    return predictions, new_y, coef

def parse_data(paths, annDir, use_reach, testing=False):
    ''' Reads and process stuff '''

    # Set this to false to generate negative examples instead of reach context annotations

    def copy_positives(data, factor, randomize = False):
        ''' Creates an augmented copy of data by factor factor '''
        ret = []
        # if not randomize:
        for i in range(factor):
            ret += copy.deepcopy(data)

        return ret

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
            vector = createFeatures(datum, tsv, annotationData)
            vectors.append(vector)
            # Generate clusters for type 2 eval
            hashes[accumulator+ix] = hash(datum)

        labels += [datum.label for datum in augmented]
        accumulator += len(augmented)

    print
    # print 'Positives: %i Negatives:%i Total:%i' % (pos, neg, pos + neg)
    print

    return labels, vectors, hashes, accumulated_data

def random(paths, annDir, testingIds, eval_type, use_reach):
    training_paths, testing_paths = [], []
    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        if pmcid in testingIds:
            testing_paths.append(path)
        else:
            training_paths.append(path)

    # training_labels, training_vectors, training_hashes, training_data = parse_data(training_paths, annDir)
    testing_labels, testing_vectors, testing_hashes, testing_data = parse_data(testing_paths, annDir, use_reach, testing=True)

    filtered_data, included = [], set()

    for datum in testing_data:
        if eval_type == EVAL1:
            filtered_data.append(datum)
        else:
            k = (datum.namespace, datum.evtIx, datum.ctxIx)
            if not k in included:
                included.add(k)
                filtered_data.append(datum)

    rv = bernoulli(1/50.)
    predictions = rv.rvs(size=len(filtered_data))
    y = [d.label for d in filtered_data]

    print "Random choice (Bernoulli p=.5)"
    labels = ["Isn't context", "Is context"]
    accuracy = metrics.accuracy_score(y, predictions)
    report = metrics.classification_report(y, predictions, target_names=labels)
    confusion = metrics.confusion_matrix(y, predictions)

    print report
    print "Accuracy: %.2f\n" % accuracy
    print "Confusion matrix:\n\t\tIsn't context\tIs context\nIsn't context\t    %i\t\t  %i\nIs context\t    %i\t\t  %i" % (confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1])

def baseline(paths, annDir, testingIds, k, eval_type, use_reach):


    training_paths, testing_paths = [], []
    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        if pmcid in testingIds:
            testing_paths.append(path)
        else:
            training_paths.append(path)

    # training_labels, training_vectors, training_hashes, training_data = parse_data(training_paths, annDir)
    testing_labels, testing_vectors, testing_hashes, testing_data = parse_data(testing_paths, annDir, use_reach, testing=True)

    filtered_data, included = [], set()

    for datum in testing_data:
        if eval_type == EVAL1:
            filtered_data.append(datum)
        else:
            k = (datum.namespace, datum.evtIx, datum.ctxIx)
            if not k in included:
                included.add(k)
                filtered_data.append(datum)

    predictions = [policy(d, k) for d in filtered_data]
    y = [d.label for d in filtered_data]

    print "Baseline policy"
    labels = ["Isn't context", "Is context"]
    accuracy = metrics.accuracy_score(y, predictions)
    report = metrics.classification_report(y, predictions, target_names=labels)
    confusion = metrics.confusion_matrix(y, predictions)

    print report
    print "Accuracy: %.2f\n" % accuracy
    print "Confusion matrix:\n\t\tIsn't context\tIs context\nIsn't context\t    %i\t\t  %i\nIs context\t    %i\t\t  %i" % (confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1])

def temp(paths, annDir, testingIds, eval_type, crossvalidation, use_reach):

    training_paths, testing_paths = [], []
    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        if pmcid in testingIds:
            testing_paths.append(path)
        else:
            training_paths.append(path)


    training_labels, training_vectors, training_hashes, training_data = parse_data(training_paths, annDir, use_reach)
    testing_labels, testing_vectors, testing_hashes, testing_data = parse_data(testing_paths, annDir, use_reach, testing=True)

    return [i for i in testing_data+training_data if i.label == 1]

def main(paths, annDir, testingIds, eval_type, crossvalidation, use_reach):
    ''' Puts all together '''

    training_paths, testing_paths = [], []
    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        if pmcid in testingIds:
            testing_paths.append(path)
        else:
            training_paths.append(path)


    training_labels, training_vectors, training_hashes, training_data = parse_data(training_paths, annDir, use_reach)
    testing_labels, testing_vectors, testing_hashes, testing_data = parse_data(testing_paths, annDir, use_reach, testing=True)


    print "General classifier"
    errors = pipeline(training_labels, training_vectors, training_hashes, testing_labels, testing_vectors, testing_hashes, eval_type, crossvalidation, training_data)

    a = training_data if crossvalidation else testing_data
    #return map(lambda e: (a[e[0]], e[1]), errors)
    return error_frame(map(lambda e: (a[e[0]], e[1]), errors), training_paths, annDir)

def error_frame(errors, training_paths, annDir):

    tsvs, annotationData = {}, {}
    for path in paths:
        pmcid = path.split(os.path.sep)[-1].split('.')[0]
        tsv = parseTSV(path)
        aData = extractAnnotationData(pmcid, annDir)
        tsvs[path] = tsv
        annotationData[path] = aData

    rows = []
    for e in errors:
        tsv = tsvs[e[0].namespace]
        aData = annotationData[e[0].namespace]
        features = createFeatures(e[0], tsv, aData)
        features['point'] = e[0]
        features['etype'] = e[1]
        features['tsv'] = tsv
        features['annotationData'] = aData

        for k in features.keys():
            if k[1] == '_':
                del features[k]

        rows.append(features)

    return pd.DataFrame(rows)


def pipeline(labels, vectors, hashes, testing_labels, testing_vectors, testing_hashes, eval_type, crossvalidation, training_data):

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


    # print "Total positive instances: %i\tTotal negative instances: %i" % (y[y == 1].shape[0], y[y == 0].shape[0])
    # Train and test a classifier
    errors = machineLearning(X, y, clusters, X_test, y_test, testing_clusters, fnames, crossvalidation, training_data)
    print

    return errors

def split_errors_by_type(errors):
    tp = errors[errors.etype == 'TP']
    fp = errors[errors.etype == 'FP']
    tn = errors[errors.etype == 'TN']
    fn = errors[errors.etype == 'FN']

    return tp, fp, tn, fn

def print_tsv_segment(element):

    ret = []

    datum, tsv, annotationData = element.point, element.tsv, element.annotationData
    points = (datum.evtIx, datum.ctxIx)
    start, end = min(points), max(points)+1


    sections, docnums = annotationData['sections'], annotationData['docnums']

    print '%s %s %s' % (datum.namespace, datum.evt, datum.ctx)
    if datum.evtIx <= datum.ctxIx:
        ret.append("EVENT ->")
    else:
        ret.append("CONTEXT ->")

    prev_docnum = None
    for i in range(start, end):

        text = tsv[i]['text']
        section = sections[i]
        docnum = docnums[i]

        if prev_docnum is not None and prev_docnum != docnum:
            ret.append('_________________________________')

        prev_docnum = docnum
        ret.append('%s: %s' % (section, text))

    if datum.evtIx <= datum.ctxIx:
        ret.append("<- CONTEXT")
    else:
        ret.append("<- EVENT")

    return ret


# Entry point
if __name__ == "__main__":
    directory = sys.argv[1]
    annDir = sys.argv[2]
    testing_ids = {s[:-1] for s in open(sys.argv[3])} if len(sys.argv) > 3 else set()
    paths = glob.glob(os.path.join(directory, '*.tsv'))
    #paths = ['/Users/enoriega/Dropbox/Context Annotations/curated tsv/PMC2063868_E.tsv']

    use_reach = False
    ev = EVAL1

    if use_reach:
        print "Using REACH's data"

    errors = main(paths, annDir, testing_ids, eval_type=ev, use_reach = use_reach, crossvalidation=False)
    # baseline(paths, annDir, testing_ids, k=7, eval_type=ev, use_reach = use_reach)
    # random(paths, annDir, testing_ids, eval_type=ev, use_reach = use_reach)
