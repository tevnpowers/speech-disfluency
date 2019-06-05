import os

import sklearn
import sklearn_crfsuite
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn_crfsuite import metrics

from classifier import sent2features, sent2labels
from load_data import load_data, get_parsed_utterance

DATA_DIRS = [
    '/corpora/LDC/LDC99T42/RAW/dysfl/dps/swbd/2',
    '/corpora/LDC/LDC99T42/RAW/dysfl/dps/swbd/3',
    '/corpora/LDC/LDC99T42/RAW/dysfl/dps/swbd/4'
]

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
