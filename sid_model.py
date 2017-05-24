from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, TimeDistributed, SimpleRNN, GRU
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
from time import sleep
import random
import sys
import h5py

path = './textdatasets/siddhartha.txt'
text = open(path).read().lower()

allowed_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', ',', '-', '.', ' ', '!', '"', "'", '(', ')','/']
allowed_chars.append('\n')
text = ''.join(filter(allowed_chars.__contains__, text))

print('corpus length:', len(text))

tensorboard = TensorBoard(log_dir='./tb_logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# split the corpus into sequences of length=maxlen
#input is a sequence of 40 chars and target is also a sequence of 40 chars shifted by one position
#for eg: if you maxlen=3 and the text corpus is abcdefghi, your input ---> target pairs will be
# [a,b,c] --> [b,c,d], [b,c,d]--->[c,d,e]....and so on
maxlen = 40
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen+1, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i+1:i +1+ maxlen])
    #if i<10 :
       # print (text[i: i + maxlen])
        #print(text[i+1:i +1+ maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences),maxlen, len(chars)), dtype=np.bool) # y is also a sequence , or  a seq of 1 hot vectors
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1

for i, sentence in enumerate(next_chars):
    for t, char in enumerate(sentence):
        y[i, t, char_indices[char]] = 1


# build the model: 2 stacked GRU
print('Build model...')
model = Sequential()
model.add(GRU(256, input_shape=(None, len(chars)), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
print ('model is made')


# train the model, output generated text after each iteration
print (model.summary())

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

for iteration in range(1, 30):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history=model.fit(X, y, batch_size=128, epochs=1, callbacks=[tensorboard])
    model.save('sid-iter-' + str(iteration) + '.h5')

    for diversity in [0.4, 0.8, 1.2]:
        print()
        print('----- diversity:', diversity)

        seed_string="siddhartha "
        sys.stdout.write(seed_string)
        #x=np.zeros((1, len(seed_string), len(chars)))
        for i in range(100):
            x=np.zeros((1, len(seed_string), len(chars)))
            #x = np.zeros((1, maxlen, len(chars)))

            for t, char in enumerate(seed_string):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            # next_index=np.argmax(preds[len(seed_string)-1])

            # print (preds[len(seed_string)-1].shape)
            # print (preds)

            next_index = sample(preds[len(seed_string)-1], diversity)

            next_char = indices_char[next_index]
            seed_string = seed_string + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()