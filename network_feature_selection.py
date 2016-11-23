from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import progressbar

# Progressbar setup
bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
])

# loading the data set
# Parameters
top_feat_no = [10, 50, 100, 200, 300, 400, 500, 600]

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

    # Perceptron Model
    pe = Perceptron(n_iter=10, eta0=10, n_jobs=-1)
    pe.fit(X_train2_sd, Y_train2)

    # Naive Bayes Classification
    nb = GaussianNB()
    nb.fit(X_train2, Y_train2)

    # Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=4)
    dt.fit(X_train2, Y_train2)

    # RandomForest Classifier
    rf = RandomForestClassifier(criterion='entropy', n_estimators=10, n_jobs=-1)
    rf.fit(X_train2, Y_train2)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train2_sd, Y_train2)

    # KNeighborsClassifier
    gpc = KNeighborsClassifier()
    gpc.fit(X_train2_sd, Y_train2)

    print("Models trained")
    print("Predicting...")

    # Prediction for Perceptron
    Y_pred_pe = pe.predict(X_test2_sd)

    # Prediction for Naive Bayes
    Y_pred_nb = nb.predict(X_test2)

    # Prediction for Decision Tree Classifier
    Y_pred_dt = dt.predict(X_test2)

    # Prediction for RandomForest Classifier
    Y_pred_rf = rf.predict(X_test2)

    # Prediction for Logistic Regression
    Y_pred_lr = lr.predict(X_test2_sd)

    # Prediction for Gaussian Process Classifier
    Y_pred_gpc = gpc.predict(X_test2_sd)

    print("**************************************************")
    print("Features selected : {}".format(f_s))

    # Accuracy, precision, recall and F1 score
    for y, x in zip([Y_pred_pe, Y_pred_nb, Y_pred_dt, Y_pred_rf, Y_pred_lr, Y_pred_gpc],
                    ['PERCEPTRON', 'NAIVE_BAYES', 'DECISION_TREE', 'RANDOM_FOREST', 'LOGISTIC_REGRESSION',
                     'KNeighborsClassifier']):
        print(x)
        print("Errors    : %d" % (Y_test2 != y).sum())
        print("Accuracy  : %.2f%%" % (accuracy_score(y, Y_test2) * 100))
        print("Precision : %.2f%%" % (precision_score(y, Y_test2) * 100))
        print("Recall    : %.2f%%" % (recall_score(y, Y_test2) * 100))
        print("F1 Score  : %.2f%% \n" % (f1_score(y, Y_test2) * 100))
    print("**************************************************")
    print("\n")
