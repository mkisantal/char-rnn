"""
  loading text, converting it to tensors of one-hot vectors,
  splitting to train/eval/test sets
  splitting sets to batches
"""

import numpy as np


class MinibatchLoader:
    def __init__(self):
        return

    def load_text(self, split_fractions=[0.8, 0.1, 0.1], batch_size=2):
        # load data
        data = open('input.txt', 'r').read()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        ix_to_char = {i: ch for i, ch in enumerate(chars)}

        # turning text to matrix of onehot vectors
        x = np.zeros([vocab_size, 0], dtype='int')
        for ch in data:
            onehot = np.zeros([vocab_size, 1], dtype='int')
            onehot[char_to_ix[ch]] = 1
            x = np.column_stack((x, onehot))

        x = x[:, :x.shape[1] - x.shape[1] % batch_size]  # remove extra-batch character
        x = np.reshape(x, (vocab_size, batch_size, -1))

        [ntrain, nvalid, _] = np.floor(data_size * np.array(split_fractions))
        ntest = data_size - ntrain - nvalid

        print(x)
        #print(ntrain, nvalid, ntest)






