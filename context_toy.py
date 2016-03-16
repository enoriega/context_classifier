import csv, glob, sys, os, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn import cross_validation, metrics
from sklearn.feature_extraction import DictVectorizer

class Datum(object):
    ''' Represents an event/candidate context mention pair '''

    def __init__(self, evtIx, ctxIx, ctx, label):
        self.evtIx = evtIx
        self.ctxIx = ctxIx
        self.ctx = ctx
        self.label = label


def extractData(tsv):
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
    for e in events:
        eid, line, ctx = e

        # Get the context line
        try:
            cLine = cLines[ctx]
            true.append(Datum(line, cLine, ctx, 1))
        except e:
            print e


        #print '%s: %s' % (k, [x[2] for x in g])

        localContext = groups[eid]

        # Pick a negative example
        # ctx2 =  getOtherContext(localContext[0], localContext)
        ctx2s = getOtherContext(line, localContext)

        if ctx2s is not None:
            for ctx2 in ctx2s:
            #     if ctx2.isdigit():
            #         print ctx2
                try:
                    cLine2 = cLines[ctx2]
                    true.append(Datum(line, cLine2, ctx2, 0))
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

    return {
        'sections':sections,
        'titles':titles
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
        #'ctxFirst':(datum.evtIx < datum.ctxIx),
        'sameLine':(datum.evtIx == datum.ctxIx),
        'ctxType':datum.ctx[0].upper(),
        #'ctxSecitonType':sectionType(sections[datum.ctxIx]),
        #'evtSecitonType':sectionType(sections[datum.ctxIx]),
        #'ctxInTitle':titles[datum.ctxIx],
        #'ctxOfOtherEvt':len({d for d in otherData if d.ctx == datum.ctx}) > 0
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

    print "Total positive instances: %i\tTotal negative instances: %i" % (y[y == 1].shape[0], y[y == 0].shape[0])
    # Train and test a classifier
    machineLearning(X, y, fnames)

# Entry point
if __name__ == "__main__":
    directory = sys.argv[1]
    annDir = sys.argv[2]
    paths = glob.iglob(os.path.join(directory, '*.tsv'))
    #paths = ['/Users/enoriega/Dropbox/Context Annotations/curated tsv/PMC2063868_E.tsv']

    main(paths, annDir)
