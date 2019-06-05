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
