import pandas as pd
import theano
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
import numpy as np

# Hyperparameters
batch_size = 128
nb_classes = 2
nb_epoch = 12
eta = 0.1

# loading the data set
df = pd.read_csv('LABTESTS_DEATH_V2.csv', header=None)
X = df.iloc[1:, 1:-2].values
Y = df.iloc[1:, -1].values  # Last column in the Dataframe contains the class label
Y = Y.astype(theano.config.floatX)  # Convert the datatype to float32 to use the GPU

# Split train test, with 90% of data used for training ang 10% for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# Standardization of data
sc = StandardScaler()
sc.fit(X_train)
X_train_sd = sc.transform(X_train)
X_test_sd = sc.transform(X_test)
X_train_sd = X_train_sd.astype('float32')
X_test_sd = X_test_sd.astype('float32')
X_test_sd = np.reshape(X_test_sd, (X_test_sd.shape[0], 33, 22))
X_train_sd = np.reshape(X_train_sd, (X_train_sd.shape[0], 33, 22))

# Convert the value in the label to a one hot vector
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(X_train_sd.shape[1], X_test_sd.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(X_train_sd, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test_sd, Y_test))

score = model.evaluate(X_test_sd, Y_test, verbose=0)

print('Test loss    : %.2f%%' % (score[0] * 100))
print('Test accuracy : %.2f%%' % (score[1] * 100))
