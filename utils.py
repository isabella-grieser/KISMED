

def calc_data_amount(labels):
    #data has 4 labels: N, O, ~, A
    o = sum(1 for l in labels if l == 'O')
    a = sum(1 for l in labels if l == 'A')
    n = sum(1 for l in labels if l == 'N')
    char = sum(1 for l in labels if l == '~')
    print(f'total data amount: {len(labels)}')
    print(f'data amount: label O: {o}; label N: {n}; label A: {a}; label ~: {char}')
