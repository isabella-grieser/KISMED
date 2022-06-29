from wettbewerb import load_references

label_mapping = {'N': 0, 'A': 1, 'O': 2, '~': 3}
reverse_map = dict((v, k) for k, v in label_mapping.items())

def data_amount(labels):
    o = sum(1 for l in labels if l == 'O')
    a = sum(1 for l in labels if l == 'A')
    n = sum(1 for l in labels if l == 'N')
    char = sum(1 for l in labels if l == '~')
    return o, a, n, char



def calc_data_amount(labels):
    # data has 4 labels: N, O, ~, A
    o, a, n, char = data_amount(labels)

    print(f'total data amount: {len(labels)}')
    print(f'all label types: {set(labels)}')
    print(f'data amount: label O: {o}; label N: {n}; label A: {a}; label ~: {char}')


def get_all_data():
    """get the data provided + all data found by us"""
    ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="data/training")
    cinc_leads, cinc_labels, _, _ = load_references(folder="data/others/CINC/fs_300")
    cpsc_leads, cpsc_labels, _, _ = load_references(folder="data/others/CPSC/fs_300")
    cu_leads, cu_labels, _, _ = load_references(folder="data/others/CU_SPH/fs_300")

    ecg_leads.extend(cinc_leads)
    ecg_leads.extend(cpsc_leads)
    ecg_leads.extend(cu_leads)

    ecg_labels.extend(cinc_labels)
    ecg_labels.extend(cpsc_labels)
    ecg_labels.extend(cu_labels)

    return ecg_leads, ecg_labels, fs, ecg_names

def load_test_references():
    """test data which can be used when given data was used for testing"""
    ecg_leads, ecg_labels, fs, ecg_names = load_references(folder="data/others/CINC/test")

    return ecg_leads, ecg_labels, fs, ecg_names


def labels_to_encodings(labels):
    return [label_mapping[l] for l in labels]


def encodings_to_labels(encodings):
    return [reverse_map[e] for e in encodings]
