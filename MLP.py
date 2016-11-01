"""
Train a MLP network to predict the mortality of patients in ICU
"""

import pandas as pd
import theano
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Hyperparameters
batch_size = 50
nb_classes = 2
nb_epoch = 12
eta = 0.1

# loading the data set
df = pd.read_csv('LABTESTS_DEATH.csv', header=None)
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
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Convert the value in the label to a one hot vector
Y_train = np_utils.to_categorical(Y_train, 2)
Y_test = np_utils.to_categorical(Y_test, 2)

# Keras Model
model = Sequential()
model.add(Dense(100, input_dim=726, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, init='uniform'))
model.add(Activation('softmax'))
model.summary()

# Optimizer
sgd = SGD(lr=0.1)

model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss    : %.2f%%' % (score[0] * 100))
print('Test accuracy : %.2f%%' % (score[1] * 100))
