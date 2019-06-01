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

# These don't seem to be used in POS-tagged dataset
OVERLAP_MARKER = '#'
CONTINUATION_MARKER = '--'
COMPLETE_MARKER = '/'
INCOMPLETE_MARKER = '-/'
#

# Special tokens that stand on their own (with no POS tag associated)
STANDALONE_TOKENS = set([
    ASIDE_START,
    COORDINATING_START,
    DISCOURSE_START,
    EDITING_START,
    FILLER_START,
    NON_SENTENCE_END,
    RESTART_BEGIN,
    RESTART_END,
    REPAIR_MARKER,
])

IGNORE_TOKENS = [
    'E_S',
    'N_S',
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

symbol_pattern = re.compile(r'^\W+$')

# Speaker Labels
SPEAKER_A = 'A'
SPEAKER_B = 'B'


class Token:
    def __init__(self, token, pos=None, is_dysfl_markup=False):
        if is_dysfl_markup:
            if token not in STANDALONE_TOKENS:
                raise Exception(f'Disfluency markup token {token} not recognized')

        self.token = token
        self.pos = pos
        self.is_dysfl_markup = is_dysfl_markup

        self.is_symbol = symbol_pattern.match(token) is not None

        # How many layers into a nested edit this is, 0 if not inside an edit
        self.edit_depth = 0

        self.is_begin_edit = False
        self.is_interruption_point = False

    @property
    def is_inside_edit(self):
        return self.edit_depth > 0

    def __str__(self):
        dysfl_tag = ''
        if self.is_begin_edit:
            if self.is_interruption_point:
                dysfl_tag = 'BE-IP'
            else:
                dysfl_tag = 'BE'
        elif self.is_interruption_point:
            dysfl_tag = 'IP'
        elif self.is_inside_edit:
            dysfl_tag = 'IE'

        pos = f'/{self.pos}' if self.pos else ''
        return f'{self.token} {dysfl_tag}'.strip()


class Sequence:
    def __init__(self):
        self.sequence = []

        # Track multiple levels of edits
        self.edit_depth = 0

        # If we find a begin or end marker, we'll attach it to the token after
        self.next_is_begin = False
        self.prev_token = None

    def add_token(self, token):
        if not isinstance(token, Token):
            raise Exception('Give me a Token object pretty please')

        if token.is_dysfl_markup:
            if token.token == RESTART_BEGIN:
                self.edit_depth += 1
                self.next_is_begin = True
            elif token.token == REPAIR_MARKER:
                # IPs get attached to the previous token
                self.prev_token.is_interruption_point = True

                # We end checking for edits at the IP
                self.edit_depth -= 1

            # TODO handle the other kinds of markup here
            # Don't add to the sequence since they're metadata
            return

        if token.is_symbol and self.edit_depth > 0:
            # Ignore symbols inside edits
            return

        if self.next_is_begin:
            token.is_begin_edit = True
            self.next_is_begin = False

        if self.edit_depth > 0:
            token.edit_depth = self.edit_depth

        self.sequence.append(token)

        self.prev_token = token

    def __str__(self):
        return ' '.join(str(t) for t in self.sequence)


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
    tokens = get_tokens(utterance)
    print(tokens)
    return parse_tokens(0, tokens)[1]


def parse_tokens(i, tokens):
    sequence = Sequence()
    if tokens:
        while i < len(tokens):
            token = tokens[i]

            try:
                if token in IGNORE_TOKENS:
                    # Don't bother with it
                    pass
                elif token in STANDALONE_TOKENS:
                    sequence.add_token(Token(token, is_dysfl_markup=True))
                else:
                    word, pos = token.split('/')
                    sequence.add_token(Token(word, pos=pos))
            except Exception:
                print(f'Failed to parse "{token}"')
                raise

            i += 1

    return i, sequence


if __name__ == '__main__':
    input_file = 'data/sw2005_trimmed.dps'
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
        print('-' * 20)
        print(get_parsed_utterance(utterance))
