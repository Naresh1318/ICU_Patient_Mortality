import numpy as np
import pandas as pd
import theano
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

K.set_image_dim_ordering('th')

batch_size = 128
nb_classes = 2
nb_epoch = 12

# loading the data set
df = pd.read_csv('LABTESTS_DEATH_V2.csv', header=None)
X = df.iloc[1:, 1:-2].values
Y = df.iloc[1:, -1].values
Y = Y.astype(theano.config.floatX)

# Split train test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_sd = sc.transform(X_train)
X_test_sd = sc.transform(X_test)
X_train_sd = X_train_sd.astype('float32')
X_test_sd = X_test_sd.astype('float32')
X_test_sd = np.reshape(X_test_sd, (X_test_sd.shape[0], 1, 33, 22))
X_train_sd = np.reshape(X_train_sd, (X_train_sd.shape[0], 1, 33, 22))

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1, 33, 22)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, init='uniform'))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

history = model.fit(X_train_sd, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test_sd, Y_test))
score = model.evaluate(X_test_sd, Y_test, verbose=0)
print('Test loss    : %.2f%%' % (score[0] * 100))
print('Test accuracy : %.2f%%' % (score[1] * 100))
