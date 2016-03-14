''' Checks context's tsv files for integrity and prints warings and errors at file level
    and line level '''

import sys
import csv
import re

NUM_COLS = 7
UNDEFINED_CID = 'NA'
HEADER = 'Line #ContextsAnnotationsRelationsTextComments'


def check_header(line, cols, errors, warnings):
    ''' Checks whether the current line is the header '''
    # Is it a header?
    if ''.join(cols) == HEADER:
        errors.append((line, "Header row wasn't removed"))

def check_num_cols(line, cols, errors, warnings):
    ''' Veryfies the number of columns in the line '''
    # Number of columns
    num = len(cols)
    if num != 7:
        errors.append((line, "Should've 7 cols, has %i" % num))

def check_col_formats(line, cols, errors, warnings):
    ''' Checks that the format of the columns is addecuate '''
    # Check col 1
    try:
        int(cols[0])
    except ValueError:
        warnings.append((line, 'Col 1 (index) is not a valid number'))

    # Check for undefined context ids
    if cols[1] == UNDEFINED_CID:
        warnings.append((line, 'Context ID not specified'))

    # Check for unexpanded ranges:
    if '-' in cols[3] or '-' in cols[4]:
        errors.append((line, 'Unexpanded annotation interval'))
    else:
        # If everything is expanded, make sure it is well formed
        tbann = cols[3].replace(' ', '')
        if tbann != '':
            tbann_match = re.match(r'^[ecst][0-9]+(?:,[ecst][0-9]+)*$', tbann, re.I)

            if tbann_match is None:
                errors.append((line, 'Text-bound annotations are not well-formed'))

        linked_context = cols[4].replace(' ', '')
        if linked_context != '':
            lctx_match = re.match(r'^[cst][0-9]+(?:,[cst][0-9]+)*$', linked_context, re.I)

            if lctx_match is None:
                errors.append((line, 'Associated contexts are not well-formed'))

    # Check that the line's text is not empty
    if cols[6].replace(' ', '') == '':
        warnings.append((line, 'Line text is empty'))

def check_transcriptions(line, cols, errors, warnings):
    ''' Checks that there is a transcription for each annotation and viceversa '''

    # Check text bound annotations:
    tbann = cols[3].replace(' ', '')
    ctx_trans = cols[2].replace(' ', '')
    ctx_ids = cols[1].replace(' ', '')
    evt_trans = cols[5].replace(' ', '')

    if tbann == '':
        if ctx_trans != '':
            errors.append((line, 'Context transcription present but text-bound annotations missing'))
        if evt_trans != '':
            errors.append((line, 'Event transcription present but text-bound annotations missing'))
        if ctx_ids != '':
            errors.append((line, 'Context id present but text-bound annotations missing'))
    else:
        if 'e' in tbann:
            if evt_trans == '':
                errors.append((line, 'Event transcription missing'))
        if re.match(r'[cst]', tbann, re.I) is not None:
            if ctx_trans == '':
                errors.append((line, 'Context transcription missing'))

    if ctx_trans != '' and ctx_ids == '':
        errors.append((line, 'Context transcription present but id missing'))
    elif ctx_ids != '' and ctx_trans == '':
        errors.append((line, 'Context id present but transcription missing'))

CHECKERS = [check_header, check_num_cols, check_col_formats, check_transcriptions]

def main(paths):
    ''' Main execution loop '''
    errors = []
    warnings = []
    for path in paths:
        print "Analizing %s ..." % path
        with open(path) as fhd:
            reader = csv.reader(fhd, delimiter='\t')
            # Iterate over the rows and apply all checkers
            for lix, row in enumerate(reader):
                for checker in CHECKERS:
                    checker(lix+1, row, errors, warnings) # Added one to the line index for convenience

        # Print summary
        for inx, err in enumerate(errors):
            print 'E%i, line %i:\t%s' % (inx, err[0], err[1])

        for inx, war in enumerate(warnings):
            print 'W%i, line %i:\t%s' % (inx, war[0], war[1])

if __name__ == '__main__':
    main(sys.argv[1:])
