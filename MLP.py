"""
Train a MLP network to predict the mortality of patients in ICU
"""

import pandas as pd
import theano
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# Hyperparameters
batch_size = 128
nb_classes = 2
nb_epoch = 12
eta = 0.1
top_feat_no = [10, 50, 100, 200, 300, 400, 500, 600, 726]
mlp_loss = []
mlp_acc = []
print("Loading data...")

# loading the data set
df = pd.read_csv('LABTESTS_DEATH_V2.csv', header=None)
X = df.iloc[1:, 1:-2].values
Y = df.iloc[1:, -1].values
Y = Y.astype(np.float)

# Read the D_LABITEMS file to find the test names corresponding to the top_test_ids
df_labitem = pd.read_csv('D_LABITEMS.csv')

# Split train test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

print("Data loaded!!!")

# Used to get the feature labels from the DataFrame
feat_labels = df.columns[1:-2]
# Train RandomForest Classifier to know the feature importance
forest = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)
forest.fit(X_train, Y_train)
# Get the feature importance from the RandomForestClassifier, has a high value for important features
importance = forest.feature_importances_
# Get the indices of the important features, 1 is added because the indices start at 0
indices = np.argsort(importance) + 1

for f_s in top_feat_no:
    # Select the best indices columns from the DataFrame,
    # last top_feat_no are chosen since the indices are sorted in the ascending order
    df_ = df[indices[-f_s:]]
    # Get only the test values from the DataFrame
    X2 = df_.iloc[1:, :].values
    # Copy the target list
    Y2 = Y[:]
    Y2 = Y2.astype(theano.config.floatX)  # Convert the datatype to float32 to use the GPU

    # To find the top tests which predict ICU death
    test_ids = df.iloc[0, :].values
    # Get the test ids for the top tests
    top_test_ids = test_ids[indices[-f_s:]]
    # Arrange it in descending order
    top_test_ids = top_test_ids[::-1]

    print(top_test_ids)
    # Print the Column Names
    print('SI ITEMID %-*s %-*s %-*s %-*s' % (30, 'LABEL', 30, 'FLUID', 30, 'CATEGORY', 30, 'LOINC_CODE'))
    # Get the names of the tests and make a table
    for c, i in enumerate(top_test_ids):
        # Find the rows which contain the ITEMID as one of the top_test_ids
        test = df_labitem[df_labitem['ITEMID'] == i].values
        print('%-*d) %d %-*s %-*s %-*s %-*s' % (3, c + 1, test[0, 1], 30, test[0, 2], 30, test[0, 3], 30, test[0, 4],
                                                30, test[0, 5]))

    print("Features Selected!!!")
# Split the data into training and testing sets
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.1, random_state=0)

    # Standardization
    sc = StandardScaler()
    sc.fit(X_train2)
    X_train2_sd = sc.transform(X_train2)
    X_test2_sd = sc.transform(X_test2)

    print("Data loaded...")
    X_train2_sd = X_train2_sd.astype('float32')
    X_test2_sd = X_test2_sd.astype('float32')

    # Convert the value in the label to a one hot vector
    Y_train2 = np_utils.to_categorical(Y_train2, 2)
    Y_test2 = np_utils.to_categorical(Y_test2, 2)

    # Keras Model
    model = Sequential()
    model.add(Dense(100, input_dim=f_s, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, init='uniform'))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='mean_squared_error',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    history = model.fit(X_train2_sd, Y_train2,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test2_sd, Y_test2))

    score = model.evaluate(X_test2_sd, Y_test2, verbose=0)

    mlp_acc.append(score[1] * 100)
    mlp_loss.append(score[0] * 100)

    print('Test loss     : %.2f%%' % (score[0] * 100))
    print('Test accuracy : %.2f%%' % (score[1] * 100))
# Plotting Graphs
# x-axis : without PCA,
per_x_axis = [10, 50, 100, 200, 300, 400, 500, 600, 726]
plt.subplot(1, 2, 1)
plt.plot(per_x_axis, mlp_acc, marker='o', color='blue', label='Accuracy')
plt.ylim([90.0, 95.0])
plt.title('{} Feature Selection'.format('Multilayer Perceptron'))
plt.xlabel('Features Selected')
plt.ylabel('Accuracy percentage')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(per_x_axis, mlp_loss, label='loss')
plt.xlabel('Features Selected')
plt.ylabel('loss percentage')
plt.title('loss plot')

plt.show()
