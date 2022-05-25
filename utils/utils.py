label_mapping = {'N': 0, 'A': 1, 'O': 2, '~': 3}
reverse_map = dict((v, k) for k, v in label_mapping.items())


def calc_data_amount(labels):
    # data has 4 labels: N, O, ~, A
    o = sum(1 for l in labels if l == 'O')
    a = sum(1 for l in labels if l == 'A')
    n = sum(1 for l in labels if l == 'N')
    char = sum(1 for l in labels if l == '~')

    print(f'total data amount: {len(labels)}')
    print(f'all label types: {set(labels)}')
    print(f'data amount: label O: {o}; label N: {n}; label A: {a}; label ~: {char}')


def labels_to_encodings(labels):
    return [label_mapping[l] for l in labels]


def encodings_to_labels(encodings):
    return [reverse_map[e] for e in encodings]
