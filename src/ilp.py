import operator

def run_ilp_program(probability_distributions):
    labels = []
    for token_distribution in probability_distributions:
        label = max(token_distribution.items(), key=operator.itemgetter(1))[0]
        labels.append(label)
    return labels