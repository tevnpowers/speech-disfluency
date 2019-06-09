import operator
import pulp

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

def optimize(probability_distributions):
    labels = []
    # for token_distribution in probability_distributions:
    #     label = max(token_distribution.items(), key=operator.itemgetter(1))[0]
    #     labels.append(label)
    # return labels

    lp = pulp.LpProblem("Classify", pulp.LpMaximize)

    # Collapse the entire sentence into a single dictionary
    # Append each tag with _i where i is its index in the sentence

    # Generate the tag name based on index
    tag_name = lambda t, i: '{}_{}'.format(t, i)

    probs = {}
    for i, tags in enumerate(probability_distributions):
        for tag, prob in tags.items():
            probs[tag_name(tag, i)] = prob
    sent_length = len(probability_distributions)
    num_tags = len(probs)

    lp_vars = pulp.LpVariable.dicts("tags", probs, lowBound=0, cat="Integer")

    # Get a variable from lp_vars based on tag name and index
    get_var = lambda t, i: lp_vars[tag_name(t, i)]

    # Set the objective function. maximixe the sum of all probabilities across the sentence
    lp += pulp.lpSum([probs[tag] * lp_vars[tag] for tag in probs.keys()])

    # Set constaints for each token in the sentence
    for i in range(sent_length):
        be = get_var('BE', i)
        be_ip = get_var('BE-IP', i)
        ip = get_var('IP', i)
        ie = get_var('IE', i)
        o = get_var('O', i)

        # There can only be one tag for every token
        lp += pulp.lpSum([be, be_ip, ip, ie, o]) == 1

        if i == 0:
            # First tag can only be a BE, BE-IP, or O
            lp += pulp.lpSum([be, be_ip, o]) == 1

        if i == sent_length - 1:
            # Last tag can only be a BE-IP, IP, or O
            lp += pulp.lpSum([be_ip, ip, o]) == 1

        if i > 0:
            prev_be = get_var('BE', i - 1)
            prev_be_ip = get_var('BE-IP', i - 1)
            prev_ip = get_var('IP', i - 1)
            prev_ie = get_var('IE', i - 1)
            prev_o = get_var('O', i - 1)

            # BE-IP|IP|O -> BE
            lp += pulp.lpSum([be, -1 * prev_be_ip, -1 * prev_ip, -1 * prev_o]) <= 0

            # BE-IP|IP|O -> BE-IP
            lp += pulp.lpSum([be_ip, -1 * prev_be_ip, -1 * prev_ip, -1 * prev_o]) <= 0

            # BE|BE-IP|IP|IE -> IP
            lp += pulp.lpSum([ip, -1 * prev_be, -1 * prev_be_ip, -1 * prev_ip, -1 * prev_ie]) <= 0

            # BE|BE-IP|IP|IE -> IE
            lp += pulp.lpSum([ie, -1 * prev_be, -1 * prev_be_ip, -1 * prev_ip, -1 * prev_ie]) <= 0

            # BE-IP|IP|O -> O
            lp += pulp.lpSum([o, -1 * prev_be_ip, -1 * prev_ip, -1 * prev_o]) <= 0

            # Can't go BE to BE
            lp += pulp.lpSum([-1 * be, -1 * prev_be, -1 * prev_ip, -1 * prev_o]) >= -1

    lp.solve()

    # Now we gotta rebuild the tag sequence from the results
    for i in range(sent_length):
        dist = {
            'BE': get_var('BE', i).varValue,
            'BE-IP': get_var('BE-IP', i).varValue,
            'IP': get_var('IP', i).varValue,
            'IE': get_var('IE', i).varValue,
            'O': get_var('O', i).varValue,
        }

        label = max(dist.items(), key=operator.itemgetter(1))[0]
        labels.append(label)

    return labels

def run_ilp_program(predictions, flatten=True):
    y_pred = []
    for sentence_distributions in predictions:
        if flatten:
            y_pred += optimize(sentence_distributions)
        else:
            y_pred.append(optimize(sentence_distributions))
    return y_pred
