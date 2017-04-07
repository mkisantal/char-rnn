"""
  loading text, converting it to tensors of one-hot vectors,
  splitting to train/eval/test sets
  splitting sets to batches
"""

import numpy as np


class MinibatchLoader:
    def __init__(self):
        self.batch_pointers = [0, 0, 0]
        self.batch_offset = [0, 0, 0]
        self.split_size = [0, 0, 0]
        self.char_to_ix = {}
        self.ix_to_char = {}
        self.data = np.array([])
        return

    def load_text(self, split_fractions=[0.8, 0.1, 0.1], batch_size=2):
        # load data
        text = open('input.txt', 'r').read()
        chars = list(set(text))
        text_size, vocab_size = len(text), len(chars)
        print('data has %d characters, %d unique.' % (text_size, vocab_size))
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}

        # turning text to matrix of onehot vectors
        data = np.zeros([vocab_size, 0], dtype='int')
        for ch in text:
            onehot = np.zeros([vocab_size, 1], dtype='int')
            onehot[self.char_to_ix[ch]] = 1
            data = np.column_stack((data, onehot))

        print(data)
        batch_num = data.shape[1] // batch_size
        data = data[:, :(batch_num * batch_size)]  # remove extra-batch character
        data = np.swapaxes(data, 0, 1)
        data = np.reshape(data, (batch_num, -1, vocab_size))
        data = np.swapaxes(data, 1, 2)
        self.data = data

        [ntrain, nvalid, _] = np.floor(batch_num * np.array(split_fractions))
        ntest = batch_num - ntrain - nvalid
        self.split_size = [ntrain, nvalid, ntest]
        print("split sizes:")
        print(self.split_size)
        self.batch_offset = [0, ntrain, ntrain + nvalid]

    def next_batch(self, split_index):
        minibatch = self.data[self.batch_pointers[split_index] + self.batch_offset[split_index], :, :]
        self.batch_pointers[split_index] += 1
        if self.batch_pointers[split_index] == self.split_size[split_index]:
            self.batch_pointers[split_index] = 0
        return minibatch






