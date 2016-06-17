''' Regrounds the curated tsv files with an "ids" text file '''
import csv
import sys
import os
import copy

def reground(ids, row):
    ''' Regrounds a single row '''

    ret = copy.copy(row)
    ctxTxt = row['ctxText'].strip()

    if ctxTxt != '':
        tokens = ctxTxt.split(',')
        new_ids = []
        for t in tokens:
            t = t.strip()
            assigned = False
            for k in ids.keys():
                if k == t:
                    new_ids.append(ids[k])
                    assigned = True
                    break
            if not assigned:
                print "Error finding %s in %s" % (t, ctxTxt)


        ret['ctxId'] = ','.join(new_ids)

    return ret

def main(ids_path, tsv_path):
    ''' Prints the tsv with the appropriate ids '''

    columns = ['ix', 'ctxId', 'ctxText', 'tbAnn', 'ctx', 'evtTxt', 'text']

    # Build the dictionary
    with open(ids_path) as f:
        ids = {k:v for k, v in [l.strip().split('\t') for l in f]}

    # Read the tsv file
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=columns)
        rows = [r for r in reader]

    new_rows = [reground(ids, row) for row in rows]

    with open(tsv_path + '_updated', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=columns)
        writer.writerows(new_rows)

if __name__ == '__main__':
    ids_path = sys.argv[1]
    tsv_path = sys.argv[2]
    main(ids_path, tsv_path)
