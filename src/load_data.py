import sys
import os
import re

# Input Data Annotations
DIALOGUE_START = '============================================================================='
ASIDE_START = '{A'
COORDINATING_START = '{C'
DISCOURSE_START = '{D'
EDITING_START = '{E'
FILLER_START = '{F'
NON_SENTENCE_END = '}'
RESTART_BEGIN = '['
RESTART_END = ']'
REPAIR_MARKER = '+'
OVERLAP_MARKER = '#'
CONTINUATION_MARKER = '--'
COMPLETE_MARKER = '/'
INCOMPLETE_MARKER = '-/'
IGNORE_TOKENS = [
    OVERLAP_MARKER,
    CONTINUATION_MARKER,
    COMPLETE_MARKER,
    INCOMPLETE_MARKER
]
NON_SENTENCE_DICT = {
    ASIDE_START: '<O>',
    COORDINATING_START: '<O>',
    DISCOURSE_START: '<D>',
    EDITING_START: '<E>',
    FILLER_START: '<F>'
}

# Output Data Annotations
BEGINNING_EDIT = '<BE>'
INSIDE_EDIT = '<IE>'
INTERUPTION_POINT = '<IP>'
BEGINNING_INTERUPTION = '<BE-IP>'
OUTSIDE = '<O>'

# Regex pattern for speaker info
pattern = re.compile(r'^Speaker[AB][0-9]+/SYM \./\.')

# Speaker Labels
SPEAKER_A = 'A'
SPEAKER_B = 'B'


def get_tokens(text):
    return text.split()


def get_speaker_span(text):
    match = re.match(pattern, text)
    if match:
        return match.span()


def print_utterances(utterances):
    for utterance in utterances:
        print(utterance)
        print('*'*25)


def add_to_utterances(utterances, speech):
    speech = speech.strip()
    if utterances:
        previous_utterance = utterances[-1]
        if previous_utterance.endswith(CONTINUATION_MARKER) or \
           previous_utterance.endswith(REPAIR_MARKER):
            utterances[-1] += ' ' + speech
        else:
            utterances.append(speech)
    else:
        utterances.append(speech)


def get_parsed_utterance(utterance):
    print(utterance)
    tokens = utterance.split()
    print(tokens)
    return parse_tokens(0, tokens)[1]


def parse_tokens(i, tokens, status=OUTSIDE):
    aligned_utterance = ''
    if tokens:
        while i < len(tokens):
            if tokens[i] == RESTART_BEGIN:
                i, nested_utterance = parse_tokens(i+1, tokens, INSIDE_EDIT)
                aligned_utterance += nested_utterance
            elif tokens[i] in NON_SENTENCE_DICT.keys():
                i, nested_utterance = parse_tokens(i+1, tokens, NON_SENTENCE_DICT[tokens[i]])
                aligned_utterance += nested_utterance
            elif tokens[i] == NON_SENTENCE_END or tokens[i] == RESTART_END:
                break
            elif tokens[i] in IGNORE_TOKENS:
                i += 1
                continue
            else:
                aligned_utterance += tokens[i] + ' ' + status + ' '
            i += 1

    return i, aligned_utterance


if __name__ == '__main__':
    input_file = 'data/sw2005.dps'
    content = []
    with open(input_file) as input_data:
        content = input_data.readlines()

    total_lines = len(content)
    i = 0
    read_dialogue = False

    speaker_a_utterances = []
    speaker_b_utterances = []
    while i < total_lines:
        line = content[i].strip()
        if line.strip() == DIALOGUE_START:
            read_dialogue = True
        elif read_dialogue:
            span = get_speaker_span(line)
            if span:
                speaker = line[7]  # e.g. "SpeakerB2"
                speech = line[span[1]:]
                i += 1
                while line and i < total_lines:
                    line = content[i].strip()
                    speech += ' ' + line
                    i += 1

                speech = speech.strip()

                if speaker == SPEAKER_A:
                    add_to_utterances(speaker_a_utterances, speech)
                elif speaker == SPEAKER_B:
                    add_to_utterances(speaker_b_utterances, speech)
                else:
                    print('error, speaker {}'.format(speaker))
                continue
        i += 1

    total_utterances = speaker_a_utterances + speaker_b_utterances
    for utterance in total_utterances:
        print(get_parsed_utterance(utterance))
