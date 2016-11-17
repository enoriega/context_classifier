# Python 2.7.12
# Reads a curated TSV file and attempts to anchor context and event mentions
# to a text interval.

# Future features
from __future__ import division, print_function

import argparse
import os.path
import csv
import numpy as np
from weighted_levenshtein import lev

# Using the argparse.Namespace class
Namespace = argparse.Namespace

# Configuration
Config = Namespace()
# If true, will correct 1-indexed line numbers in TSVs to 0-indexed ones
Config.zero_index_line = True
Config.event_triggers = {
    'phosphoryl', 'ubiquitina', 'hydroxylat', 'ribosylat', 'express',
    'farnesylat', 'complex', 'acetylat', 'glycosylat', 'methylat',
    'sumoylat', 'activat', 'regulat', 'translocat', 'transfe', 'promot',
    'mediat', 'interact', 'respon', 'inhibit', 'transcri', 'bind', 'block',
    'supress', 'induc', 'increa', 'associat', 'antagon', 'hamper', 'abolish',
    'synthes', 'activit', 'amplifi', 'bound', 'recruit', 'stimulat',
    'dissociat', 'hydroliz', 'contact', 'produc', 'potent', 'decreas',
    'immunoprecip', 'specificit', 'downstream', 'upstream'
}
Config.debug = False

# Collated errors
Errors = []


# Utility functions
def ed_align(line_num, free_text, sentence_text):
    """
    Uses the weighted_levenshtein package to find the token intervals (
    1-indexed) from sentence_text that best fit the free_text supplied.
    :param line_num: For marking output
    :param free_text:
    :param sentence_text:
    :return: (tuple) start_interval, end_interval
    """
    free_text_tokens = free_text.split()
    sentence_tokens = sentence_text.split()

    # Edit Distance costs -- We want to penalise most substitutions heavily,
    # but make uppercase-lowercase substitutions fairly cheap. We also want
    # to penalise additions more heavily than deletions (since the sentence
    # text is lemmatised)
    insert_costs = np.full(128, 2, dtype=np.float64)
    delete_costs = np.full(128, 1, dtype=np.float64)
    substitute_costs = np.full((128, 128), 2, dtype=np.float64)
    for letter in set('abcdefghijklmnopqrstuvwxyz'):
        substitute_costs[ord(letter), ord(letter.upper())] = 0.5
        substitute_costs[ord(letter.upper()), ord(letter)] = 0.5

    # Iterate over sentence_tokens, checking the ED of the next N tokens,
    # where N is the number of free_text_tokens.
    best_start = -1
    best_end = -1
    best_cost = -1
    for start_idx in range(0, len(sentence_tokens)):
        if start_idx + len(free_text_tokens) > len(sentence_tokens):
            # No more words left to check
            break

        # Get the combined ED for the next N tokens
        total_cost = 0
        for context_idx in range(0, len(free_text_tokens)):
            word_cost = lev(free_text_tokens[context_idx],
                            sentence_tokens[start_idx + context_idx],
                            insert_costs=insert_costs,
                            delete_costs=delete_costs,
                            substitute_costs=substitute_costs)
            d_print("{} against {}: {}"
                    "".format(free_text_tokens[context_idx],
                              sentence_tokens[start_idx + context_idx],
                              word_cost))
            total_cost += word_cost

        # Check against global best
        if best_cost == -1 or total_cost < best_cost:
            best_start = start_idx
            best_end = start_idx + len(free_text_tokens) - 1
            best_cost = total_cost
            d_print("Found better alignment: {}, {}, {}"
                    "".format(best_start, best_end, best_cost))

    if best_start == -1 or best_end == -1:
        # Something happened -- We didn't even get to check a single token
        e_print("[ERROR] Line {}: Failed to find context token(s) '{}' in "
                "sentence '{}'."
                "".format(line_num, free_text_tokens, sentence_tokens))
        return -1, -1
    else:
        # Report and return 1-indexed intervals
        start_interval = best_start + 1
        end_interval = best_end + 1
        sentence_match = " ".join(
            sentence_tokens[best_start:best_start + len(free_text_tokens)]
        )
        print("Line {}: Found best match for context '{}' at interval {}-{}; "
              "match was '{}'."
              "".format(line_num, free_text,
                        start_interval, end_interval,
                        sentence_match))
        return start_interval, end_interval


def d_print(text=''):
    if Config.debug:
        print(text)


def e_print(error):
    Errors.append(error)
    print(error)


# Main script
if __name__ == '__main__':
    # Parse arguments:
    # [1] TSV file to parse
    parser = argparse.ArgumentParser(description='Parse a full <basename>.tsv '
                                                 'file into '
                                                 '<basename>_events.tsv and '
                                                 '<basename>_contexts.tsv')
    parser.add_argument('filename')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    args = parser.parse_args()

    # Verbose (debug) mode?
    if args.verbose:
        Config.debug = True

    # Try to read main TSV
    with open(args.filename, 'rb') as fp:

        ## INIT
        # Will hold {<label> => <grounding ID>}
        context_labels = {}
        if Config.zero_index_line:
            print("Note that all line numbers referenced are 0-indexed.")
        # Will hold output for _contexts.tsv
        context_data = []
        # Will hold output for _events.tsv
        event_data = []
        basename = os.path.splitext(args.filename)[0]
        context_filename = "{}_contexts.tsv".format(basename)
        event_filename = "{}_events.tsv".format(basename)

        tsv = csv.reader(fp, delimiter='\t')
        for raw_row in tsv:
            # Parse row data
            row = Namespace()
            row.line_num = int(raw_row[0])
            if Config.zero_index_line:
                row.line_num -= 1
            row.grounding_ids = raw_row[1]
            row.grounding_texts = raw_row[2]
            # Annotation labels: S|T|C = Contexts; E = Events
            row.annotation_labels = raw_row[3]
            row.associations = raw_row[4]
            row.sentence = raw_row[6]

            # This column doesn't seem to be filled...
            row.event = raw_row[5]
            # print("Event: {}".format(row.event))

            # -----------------
            # Identify Contexts
            # -----------------
            if row.grounding_ids != "":
                # There are contexts identified on this line
                groundings = row.grounding_ids.split(",")

                # Match with annotation labels
                labels = row.annotation_labels.lower().split(",")
                free_texts = row.grounding_texts.split(",")
                for label in labels:
                    label = label.strip()
                    if not label.startswith("e"):
                        # This is a context
                        if len(groundings) == 0:
                            # Crap, we don't have enough grounding IDs to go
                            # around
                            e_print("[ERROR] Line {}: Could not find a "
                                    "grounding ID for label '{}'."
                                    "".format(row.line_num, label))
                            continue

                        this_grounding_id = groundings.pop(0).strip()
                        context_labels[label] = this_grounding_id

                        # Now that we have noted the annotation label,
                        # try to find the required info:
                        # 1. Line # (0-indexed)
                        # 2. Token interval (1-indexed)
                        # 3. Grounding ID
                        this_free_text = free_texts.pop(0).strip()

                        # Word-based minimal edit distance search through
                        # sentence text
                        match = ed_align(row.line_num,
                                         this_free_text, row.sentence)
                        this_interval = "-".join([str(x) for x in match])
                        context_data.append(
                            [row.line_num, this_interval, this_grounding_id]
                        )

                # Are there any groundings with no matching label?
                if len(groundings) > 0:
                    e_print("[ERROR] Line {}: Did not identify labels for the "
                            "following contexts: {}"
                            "".format(row.line_num,
                                      zip(groundings, free_texts)))

                    # Yes -- But we should process them for _context.tsv
                    # anyway
                    for this_grounding_id in groundings:
                        this_free_text = free_texts.pop(0).strip()
                        match = ed_align(row.line_num,
                                         this_free_text, row.sentence)
                        this_interval = "-".join([str(x) for x in match])
                        context_data.append(
                            [row.line_num, this_interval, this_grounding_id]
                        )

            # -----------------
            # Identify Events
            # -----------------
            # Our main cue to events is in the annotation label.
            # We will go through the sentence, picking out event triggers
            # sequentially as the labels pop up.
            current_idx = -1
            sentence_tokens = row.sentence.split()

            for label in row.annotation_labels.lower().split(","):
                if not label.startswith("e"):
                    # Not an event
                    continue

                # Find the next event trigger in the sentence, or report
                # failure
                found = False
                while not found:
                    current_idx += 1
                    if current_idx + 1 > len(sentence_tokens):
                        # Crap, we ran out of words to check
                        e_print("[ERROR] Line {}: No trigger found for event "
                                "'{}'."
                                "".format(row.line_num, label))
                        break

                    current_token = sentence_tokens[current_idx]
                    for trigger in Config.event_triggers:
                        if current_token.lower().startswith(trigger.lower()):
                            # Bingo, found an event
                            found = True
                            # Single token 1-indexed interval
                            this_interval = "{}-{}".format(current_idx + 1,
                                                           current_idx + 1)
                            # At the moment, these are annotation labels.  We
                            # will resolve them before writing to _events.tsv
                            this_groundings = row.associations.lower().split(",")

                            # Reporting every event trigger hit is a bit
                            # verbose... Make it debug-level output
                            d_print("Line {}: Found trigger for event '{}' -- "
                                    "'{}'."
                                    "".format(row.line_num, label,
                                              current_token))

                            event_data.append(
                                [row.line_num, this_interval, this_groundings]
                            )

    # --------------
    # Process Output
    # --------------
    # context_data is already nicely formatted.
    with open(context_filename, 'wb') as fp:
        tsv = csv.writer(fp, delimiter='\t')
        tsv.writerows(context_data)

    # event_data needs to have annotation labels matched to grounding IDs
    with open(event_filename, 'wb') as fp:
        tsv = csv.writer(fp, delimiter='\t')
        for row in event_data:
            # Go through the list of associations and match them, dropping
            # errors where necessary.
            this_associations = row[2]
            for idx in range(0, len(this_associations)):
                label = this_associations[idx]
                try:
                    this_associations[idx] = \
                        context_labels[label]
                except KeyError:
                    e_print("[ERROR] Line {}: Could not resolve annotation "
                            "label '{}'."
                            "".format(row[0], label))

            # Write
            tsv.writerow([row[0], row[1], ",".join(this_associations)])

    if len(Errors):
        print()
        print("Printing collated errors:")
        print("\n".join(Errors))
