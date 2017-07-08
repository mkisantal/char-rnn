"""
  loading text, converting it to tensors of one-hot vectors,
  splitting to train/eval/test sets
  splitting sets to batches
"""

import numpy as np


class MinibatchLoader:
    def __init__(self):
        self.char_to_ix = {}
        self.ix_to_char = {}
        self.x = np.array([])
        self.y = np.array([])
        self.text_pointer = 0
        self.batch = 10000
        self.vocab_size = 0
        self.timesteps = 0
        return

    def load_text(self, batch_size=32, timesteps=10, batch=10000):

        self.batch = batch
        self.timesteps = timesteps

        # clear previous batch
        self.x = []
        self.y = []

        text = open('input.txt', 'r').read()

        # building vocabularies
        chars = list(set(text))
        text_size, self.vocab_size = len(text), len(chars)

        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}

        # converting text to matrix of onehot vectors
        data = np.zeros([self.vocab_size, 0], dtype='int')
        for ch in text[self.text_pointer:(self.text_pointer+self.batch)]:
            onehot = self.encode(ch)
            data = np.column_stack((data, onehot))

        x = data

        # dividing data to training sequences
        batch_num = x.shape[1] // (batch_size * timesteps)

        x = x[:, :(batch_num * batch_size * timesteps)]
        x = np.swapaxes(x, 0, 1)
        x = np.reshape(x, (batch_num * batch_size, timesteps, self.vocab_size))
        self.x = x
        y = self.x[1:-(batch_size-1), 1, :]    # target is the first char in sequence from the next batch
        y = np.squeeze(y)
        self.y = y
        self.x = self.x[:-batch_size, :, :]  # no target for last batch TODO: very lossy

        self.text_pointer = (self.text_pointer + self.batch) % text_size

        return self.x, self.y

    def get_vocabsize(self):
        return self.vocab_size

    def encode(self, data):
        if self.vocab_size == 0:
            ValueError('Dictionary empty, load textfile first!')
        if len(data) == 1:
            onehot = np.zeros([self.vocab_size, 1], dtype='int')
            onehot[self.char_to_ix[data]] = 1
            return onehot
        else:
            onehot_matrix = np.zeros([self.vocab_size, 0], dtype='int')
            for ch in data:
                onehot = np.zeros([self.vocab_size, 1], dtype='int')
                onehot[self.char_to_ix[ch]] = 1
                onehot_matrix = np.column_stack((onehot_matrix, onehot))
            onehot_matrix = np.swapaxes(onehot_matrix, 0, 1)
            shaped = np.reshape(onehot_matrix, (-1, self.timesteps, self.vocab_size))
            return shaped

    def decode(self, pred_vector):
        if len(pred_vector.shape) == 2:
            if pred_vector.shape[0] == 1:
                ch = self.ix_to_char[pred_vector.argmax()]
                return ch
        else:
            text = ''
            for i in range(pred_vector.shape[1]):
                ch = self.ix_to_char[pred_vector[0, i, :].argmax()]
                text += ch
            return text
