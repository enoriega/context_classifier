import itertools as it

class ClassificationResults(object):
    def __init__(self, name, truth, predictions):
        self.name = name
        self.update(truth, predictions)

    def update(self, truth, predictions):
        self.truth = truth
        self.predictions = predictions

        # Compute the TP, FP, TN, FN
        tp, fp, tn, fn = 0, 0, 0, 0
        for t, p in it.izip(truth, predictions):
            if t == p == 1:
                tp += 1
            elif t == p == 0:
                tn += 1
            elif t != p and t == 1:
                fn += 1
            else:
                fp += 1

        self.true_positives = tp
        self.false_positives = fp
        self.true_negatives = tn
        self.false_negatives = fn

    @property
    def precision(self):
        den = float(self.true_positives + self.false_positives)
        if den == 0.:
            return 0
        else:
            return self.true_positives / den

    @property
    def recall(self):
        den = float(self.true_positives + self.false_negatives)
        if den == 0.:
            return 0
        else:
            return self.true_positives / den

    @property
    def accuracy(self):
        den = float(self.true_positives + self.false_negatives + self.false_positives + self.true_negatives)
        if den == 0.:
            return 0
        else:
            return (self.true_positives + self.true_negatives) / den

    @property
    def f1(self):
        den = 2*(self.precision+self.recall)
        if den == 0.:
            return 0
        else:
            return (self.precision*self.recall)/den

    def __repr__(self):
        return "CR %s - P:%f\tR:%f\tF1:%f" % (self.name, self.precision, self.recall, self.f1)

    def __len__(self):
        return len(self.truth)

class MicroAverage(ClassificationResults):
    def __init__(self, name, results):
        self.results = results
        truth = list(it.chain(*[r.truth for r in results]))
        predictions = list(it.chain(*[r.predictions for r in results]))

        super(MicroAverage, self).__init__(name, truth, predictions)

class MacroAverage(ClassificationResults):
    def __init__(self, name, results):
        self.name = name
        self.results = results

    @property
    def precision(self):
        p = sum([r.precision for r in self.results])
        return p/float(len(self))

    @property
    def recall(self):
        r = sum([r.recall for r in self.results])
        return r/float(len(self))

    @property
    def accuracy(self):
        a = sum([r.accuracy for r in self.results])
        return p/float(len(self))

    @property
    def f1(self):
        f = sum([r.f1 for r in self.results])
        return f/float(len(self))

    def __len__(self):
        return len(self.results)
