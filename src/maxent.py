from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.feature_extraction import DictVectorizer
from classifier import sent2features, sent2labels
from load_data import get_data, get_parsed_utterance


def get_features_and_labels(sentences):
    features = []
    labels = []
    for s in sentences:
        features += sent2features(s)
        labels += sent2labels(s)
    return features, labels

if __name__ == '__main__':
    print('Parsing annotated sentences...')
    sentences = []
    for utterance in get_data():
        sequence = get_parsed_utterance(utterance)
        if len(sequence.sequence) > 0:
            sentences.append(sequence.sequence)

    print('Creating 80-20 train-test split...')

    train_test_split = 0.8
    train_count = int(train_test_split*len(sentences))
    train_sents = sentences[:train_count]
    test_sents = sentences[train_count:]

    v = DictVectorizer()
    X_train, y_train = get_features_and_labels(train_sents)
    X_train = v.fit_transform(X_train)
    X_test, y_test = get_features_and_labels(test_sents)
    X_test = v.transform(X_test)
    print('Training MaxEnt on {} sentences'.format(len(y_train)))
    maxent = clf = LogisticRegression(random_state=0, solver='lbfgs',
                    multi_class='multinomial', max_iter=400)
    maxent.fit(X_train, y_train)

    labels = list(maxent.classes_)
    labels.remove('O')

    print('Evaluating MaxEnt on {} sentences'.format(len(y_test)))
    y_pred = maxent.predict(X_test)

    print('Recording metrics...')
    print(metrics.classification_report(y_test, y_pred, labels=labels))