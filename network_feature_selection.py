from __future__ import print_function
import progressbar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Progressbar setup
bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
])

# loading the data set
# Parameters
top_feat_no = [10, 50, 100, 200, 300, 400, 500, 600, 726]
# Perceptron
pe_error = []
pe_acc = []
pe_pre = []
pe_rc = []
pe_f1 = []
pe_v = [pe_error, pe_acc, pe_pre, pe_rc, pe_f1]

# Naive Bayes
nb_error = []
nb_acc = []
nb_pre = []
nb_rc = []
nb_f1 = []
nb_v = [nb_error, nb_acc, nb_pre, nb_rc, nb_f1]

# Decision Tree
dt_error = []
dt_acc = []
dt_pre = []
dt_rc = []
dt_f1 = []
dt_v = [dt_error, dt_acc, dt_pre, dt_rc, dt_f1]

# Random Forest
rf_error = []
rf_acc = []
rf_pre = []
rf_rc = []
rf_f1 = []
rf_v = [rf_error, rf_acc, rf_pre, rf_rc, rf_f1]

# Logistic Regression
lr_error = []
lr_acc = []
lr_pre = []
lr_rc = []
lr_f1 = []
lr_v = [lr_error, lr_acc, lr_pre, lr_rc, lr_f1]

# KNN
kn_error = []
kn_acc = []
kn_pre = []
kn_rc = []
kn_f1 = []
kn_v = [kn_error, kn_acc, kn_pre, kn_rc, kn_f1]

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
    for y, x, p in zip([Y_pred_pe, Y_pred_nb, Y_pred_dt, Y_pred_rf, Y_pred_lr, Y_pred_gpc],
                       ['PERCEPTRON', 'NAIVE_BAYES', 'DECISION_TREE', 'RANDOM_FOREST', 'LOGISTIC_REGRESSION',
                        'KNeighborsClassifier'], [pe_v, nb_v, dt_v, rf_v, lr_v, kn_v]):
        print(x)
        error = (Y_test2 != y).sum()
        p[0].append(error)
        print("Errors    : %d" % error)
        acc = accuracy_score(y, Y_test2) * 100
        p[1].append(acc)
        print("Accuracy  : %.2f%%" % acc)
        ps = precision_score(y, Y_test2) * 100
        p[2].append(ps)
        print("Precision : %.2f%%" % ps)
        rs = recall_score(y, Y_test2) * 100
        p[3].append(rs)
        print("Recall    : %.2f%%" % rs)
        f1 = f1_score(y, Y_test2) * 100
        p[4].append(f1)
        print("F1 Score  : %.2f%% \n" % f1)
    print("**************************************************")
    print("\n")
print("\n")
# Plotting Graphs
# x-axis : without PCA,
per_x_axis = [10, 50, 100, 200, 300, 400, 500, 600, 726]
for p, t in zip([pe_v, nb_v, dt_v, rf_v, lr_v, kn_v], ['PERCEPTRON', 'NAIVE_BAYES', 'DECISION_TREE', 'RANDOM_FOREST',
                                                       'LOGISTIC_REGRESSION', 'KNeighborsClassifier']):

    plt.subplot(1, 2, 1)
    plt.plot(per_x_axis, p[1], marker='o', color='blue', label='Accuracy')
    plt.plot(per_x_axis, p[2], marker='s', color='red', label='Precision')
    plt.plot(per_x_axis, p[3], marker='x', color='green', label='Recall')
    plt.plot(per_x_axis, p[4], marker='D', color='black', label='F1 Score')
    if t != 'NAIVE_BAYES':
        plt.ylim([80.0, 100.0])
    else:
        plt.ylim([10.0, 95.0])
    plt.title('{} Feature Selection'.format(t))
    plt.xlabel('Features Selected')
    plt.ylabel('Scores')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(per_x_axis, p[0], label='No. of Errors')
    plt.xlabel('Features Selected')
    plt.ylabel('Errors')
    plt.title('Error Plot')

    plt.show()
