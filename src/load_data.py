import sys
import os
import re

# Input Data Annotations
DIALOGUE_START = '============================================================================='
NON_SENTENCE_BEGIN = '{'
NON_SENTENCE_END = '}'
RESTART_BEGIN = '['
RESTART_END = ']'
REPAIR_MARKER = '+'
EDITING_MARKER = 'E'
FILLER_MARKER = 'F'
DISCOURSE_MARKER = 'D'

# Output Data Annotations
BEGINNING_EDIT = 'BE'
INSIDE_EDIT = 'IE'
INTERUPTION_POINT = 'IP'
BEGINNING_INTERUPTION = 'BE-IP'
OUTSIDE = 'O'

# Regex pattern for speaker info
pattern = re.compile(r'^@*[AB]\.[0-9]+:')

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
        if previous_utterance.endswith("--") or \
           previous_utterance.endswith("+"):
            utterances[-1] += ' ' + speech
        else:
            utterances.append(speech)
    else:
        utterances.append(speech)

if __name__ == '__main__':
    input_file = 'data/sw2005.dff'
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
                speaker = line[0]
                speech = line[span[1]:]
                i += 1
                while line and i < total_lines:
                    line = content[i].strip()
                    speech += ' ' + line
                    i += 1

                if speaker == SPEAKER_A:
                    add_to_utterances(speaker_a_utterances, speech)
                elif speaker == SPEAKER_B:
                    add_to_utterances(speaker_b_utterances, speech)
                else:
                    print('error, speaker {}'.format(speaker))
                continue
        i += 1

    print_utterances(speaker_a_utterances)
    print('-'*200)
    print_utterances(speaker_b_utterances)
