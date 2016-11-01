import numpy as np
import pandas as pd
import theano
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

K.set_image_dim_ordering('th')

batch_size = 128
nb_classes = 2
nb_epochs = 12

# Embedding dimensions.
row_hidden = 128
col_hidden = 128

# loading the data set
df = pd.read_csv('LABTESTS_DEATH.csv', header=None)
X = df.iloc[1:, 1:-2].values
Y = df.iloc[1:, -1].values
Y = Y.astype(theano.config.floatX)

# Split train test, with 90% of data used for training ang 10% for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# Standardization of data
sc = StandardScaler()
sc.fit(X_train)
X_train_sd = sc.transform(X_train)
X_test_sd = sc.transform(X_test)
X_train_sd = X_train_sd.astype('float32')
X_test_sd = X_test_sd.astype('float32')
X_test_sd = np.reshape(X_test_sd, (X_test_sd.shape[0], 33, 22, 1))
X_train_sd = np.reshape(X_train_sd, (X_train_sd.shape[0], 33, 22, 1))

# Convert the value in the label to a one hot vector
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

row, col, pixel = X_train_sd.shape[1:]

# 4D input.
x = Input(shape=(row, col, pixel))

# Encodes a row of pixels using TimeDistributed Wrapper.
encoded_rows = TimeDistributed(LSTM(output_dim=row_hidden))(x)

# Encodes columns of encoded rows.
encoded_columns = LSTM(col_hidden)(encoded_rows)

# Final predictions and model.
prediction = Dense(nb_classes, activation='softmax')(encoded_columns)
model = Model(input=x, output=prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Training.
model.fit(X_train_sd, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(X_test_sd, Y_test))

# Evaluation.
scores = model.evaluate(X_test_sd, Y_test, verbose=0)

print('Test loss    : %.2f%%' % (scores[0] * 100))
print('Test accuracy : %.2f%%' % (scores[1] * 100))
