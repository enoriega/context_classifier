''' This script converts a tsv file to the set of files used for training in reach/context '''

import csv
import sys
import os
from model.parsing import find_evt_anchors
from model import triggers

def get_context(row):
    ret = []
    ctx, tbound, contextId, evt, text, ctxGroundingIds = row[2].lower().strip(), row[3].lower().strip(), row[4].lower().strip(), row[5].lower().strip(), row[6].lower().strip(), row[1].lower().strip()

    ctx = ctx.split(',') if ctx != '' else []
    ctxGroundingIds = ctxGroundingIds.split(',') if ctxGroundingIds != '' else []
    tbound = filter(lambda x: x[0] != 'e', tbound.split(',')) if tbound != '' else []

    #assert len(tbound) == len(ctx), "Missmatching number of annotations and TB labels"

    words = text.split()

    offset = 0
    for ct, tb, ctxId in zip(ctx, tbound, ctxGroundingIds):
        # remove cell/cells
        ct = ct.replace('cells', '')
        ct = ct.replace('cell', '')
        ct.strip()

        start = end = None

        # Split the words
        c = ct.split()
        if len(c) > 0:
            first, last = c[0], c[-1]

            ix = 0


            # Find the location on text
            for ix, word in enumerate(words):
                if word == first:
                    start = offset + ix
                    offset += ix
                    break

            words = words[ix:]

            for ix, word in enumerate(words):
                if word == last:
                    end = offset + ix
                    offset += ix
                    break

            words = words[ix:]

        if start is not None and end is not None:
            ret.append((start, end, tb, ctxId))
        elif len(c) > 0:
            print "Problem finding %s in '%s'" % (' '.join(c), text)

    return ret

def transform(path):
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        rows = [r for r in reader]

    sentences = ['%i\t%s' % (ix, get_sentence(row)) for ix, row in enumerate(rows)]
    events, context = [], []

    context_cache = {}

    # First pass for context
    for ix, row in enumerate(rows):
        for ctx in get_context(row):
            start, end, cid, groudingId = ctx
            context_cache[cid] = groudingId.lower()
            context.append('%i\t%i-%i\t%s' % (ix, start, end, groudingId))


    for ix, row in enumerate(rows):
        if 'e' in row[3].lower():
            evts = get_events(row, context_cache)
            for evt in evts:
                start, end, contexts = evt
                events.append('%i\t%i-%i\t%s' % (ix, start, end, contexts))



    return sentences, events, context

def get_sentence(row):
    return row[-1]

def get_events(row, ctx_cache):
    sentence = row[-1]
    ctxIds = [c.strip().lower() for c in row[4].split(',')]
    trigger_nums = find_evt_anchors(sentence, triggers)

    groundedCtxIds = []
    for cid in ctxIds:
        try:
            i = ctx_cache[cid]
            groundedCtxIds.append(i)
        except KeyError:
            print "Problem finding %s in:\n%s" % (cid, '\t'.join(row))

    ctxString = ','.join(groundedCtxIds)

    ret = []
    for trigger in trigger_nums:
        ret.append((trigger, trigger, ctxString))

    return ret

if __name__ == '__main__':
    path = sys.argv[1] # TSV file path
    sentences, events, context = transform(path)

    # Create the output directory
    dirpath = path.split('.')[0].split(os.path.sep)[-1]

    try:
        os.mkdir(dirpath)
    except OSError:
        print 'Directory %s already exists' % dirpath

    with open(os.path.join(dirpath, 'sentences.tsv'), 'w') as f:
        for s in sentences: f.write('%s\n' % s)

    with open(os.path.join(dirpath, 'events.tsv'), 'w') as f:
        for s in events: f.write('%s\n' % s)

    with open(os.path.join(dirpath, 'context.tsv'), 'w') as f:
        for s in context: f.write('%s\n' % s)
