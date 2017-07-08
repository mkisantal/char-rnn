from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from text_loader2 import MinibatchLoader
minibatch_loader = MinibatchLoader()

# settings
timesteps = 10
mini_batch_size = 16
data_batch_size = 5000
num_data_batches = 2
epochs_on_data_batch = 2

# loading data, building vocabularies
x_train, y_train = minibatch_loader.load_text(batch_size=mini_batch_size, timesteps=timesteps, batch=data_batch_size)

data_dim = minibatch_loader.get_vocabsize()
num_classes = data_dim

# defining model
model = Sequential()
model.add(LSTM(32, input_shape=[timesteps, data_dim]))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


def test_model(rng, test_string):
    if len(test_string) > timesteps:
        test_string = test_string[-timesteps:]
    x_test = minibatch_loader.encode(test_string)
    answer = ''
    for j in range(rng):
        y_test = model.predict(x_test, batch_size=1)
        out_char = minibatch_loader.decode(y_test)
        answer += out_char
        feedback = np.reshape(y_test, (1, 1, -1))
        x_test = x_test[:, 1:, :]
        x_test = np.concatenate((x_test, feedback), 1)

    print(test_string + '|' + answer)


for i in range(num_data_batches):
    print(i)
    model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=epochs_on_data_batch, shuffle=False, verbose=0)
    x_train, y_train = minibatch_loader.load_text(batch_size=mini_batch_size, timesteps=timesteps, batch=5000)
    test_model(50, 'First Citi')
