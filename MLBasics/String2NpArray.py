import numpy as np


def str2arr(s):
    tok = []
    for t in s.strip('[]').split(';'):
        tok.append('[' + ','.join(t.strip().split(' ')) + ']')

    b = eval('[' + ','.join(tok) + ']')
    return np.array(b)
