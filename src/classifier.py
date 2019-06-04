import nltk
import os
import sklearn
import sklearn_crfsuite
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn_crfsuite import metrics

from load_data import load_data, get_parsed_utterance

DATA_DIRS = [
    '/corpora/LDC/LDC99T42/RAW/dysfl/dps/swbd/2',
    '/corpora/LDC/LDC99T42/RAW/dysfl/dps/swbd/3',
    '/corpora/LDC/LDC99T42/RAW/dysfl/dps/swbd/4'
]

def word2features(sent, i):
    # Best features from paper (tokens + POS tags):
    # Unigrams: (word_i), (word_i+1)
    # Bigrams: (word_i-1, word_i), (word_i, word_i+1)
    # Trigrams: (word_i-2, word_i-1, word_i), (word_i, word_i+1, word_i+2)
    features = {
        'bias': 1.0
    }
    minimum_index = max(i - 2, 0)
    maximum_index = min(i + 2, len(sent) - 1)

    for j in range(minimum_index, maximum_index + 1):
        if j == i - 2:
            # (word_i-2, word_i-1, word_i)
            trigram = sent[j:i+1]
            token_trigram = ' '.join([token.token for token in trigram])
            pos_trigram = ' '.join([token.pos_tag for token in trigram])
            features.update({
                'token_trigram-2': token_trigram.lower(),
                'pos_trigram-2': pos_trigram
            })
        elif j == i - 1:
            # (word_i-1, word_i)
            token_bigram = sent[j].token + ' ' + sent[i].token
            pos_bigram = sent[j].pos_tag + ' ' + sent[i].pos_tag
            features.update({
                'token_bigram-1': token_bigram.lower(),
                'pos_bigram-1': pos_bigram
            })
        elif j == i:
            # (word_i)
            features.update({
                'token': sent[i].token.lower(),
                'pos': sent[i].pos_tag
            })
        elif j == i + 1:
            # (word_i+1)
            features.update({
                'token+1': sent[i+1].token.lower(),
                'pos+1': sent[i+1].pos_tag
            })
            # (word_i, word_i+1)
            token_bigram = sent[i].token + ' ' + sent[j].token
            pos_bigram = sent[i].pos_tag + ' ' + sent[j].pos_tag
            features.update({
                'token_bigram+1': token_bigram.lower(),
                'pos_bigram+1': pos_bigram
            })
        elif j == i + 2:
            # (word_i, word_i+1, word_i+2)
            trigram = sent[i:j+1]
            token_trigram = ' '.join([token.token for token in trigram])
            pos_trigram = ' '.join([token.pos_tag for token in trigram])
            features.update({
                'token_trigram+2': token_trigram.lower(),
                'pos_trigram+2': pos_trigram
            })
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [token.dysfl_tag for token in sent]

if __name__ == '__main__':
    raw_data = []
    for directory in DATA_DIRS:
        print('Loading raw data from {}...'.format(directory))
        for filename in os.listdir(directory):
            if filename.endswith(".dps"):
                filepath = os.path.join(directory, filename)
                raw_data += load_data(filepath)

    print('Parsing annotated sentences...')
    sentences = []
    for utterance in raw_data:
        sequence = get_parsed_utterance(utterance)
        if len(sequence.sequence) > 0:
            sentences.append(sequence.sequence)

    print('Creating 80-20 train-test split...')

    train_test_split = 0.8
    train_count = int(train_test_split*len(sentences))
    train_sents = sentences[:train_count]
    test_sents = sentences[train_count:]

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    print('Training on {} sentences'.format(len(y_train)))
    print('Evaluating on {} sentences'.format(len(y_test)))

    print('Training CRF model...')
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    labels.remove('O')

    print('Executing model against test set...')
    y_pred = crf.predict(X_test)

    print('Recording metrics...')
    metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))
