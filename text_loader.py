"""
  loading text, converting it to tensors of one-hot vectors,
  splitting to train/eval/test sets
  splitting sets to batches
"""

import numpy as np


class TextLoader(object):
    def __init__(self, name):
        self.name = name

    def alma(self):
        return


def text_to_tensor():
    return


def load_text():

    # load data
    data = open('input.txt', 'r').read()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print( 'data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    x = np.zeros([vocab_size, 0], dtype='int')
    for ch in data:
        onehot = np.zeros([vocab_size, 1], dtype='int')
        onehot[char_to_ix[ch]] = 1
        x = np.c_[x, onehot]

    print(x)


