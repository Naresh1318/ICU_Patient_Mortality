from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import progressbar

# Progressbar setup
bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
])

# Parameters
pca_val = [10, 50, 100, 200, 300, 400, 500, 600]
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

# Split train test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# Standardization
sc = StandardScaler()
sc.fit(X_train)
X_train_sd = sc.transform(X_train)
X_test_sd = sc.transform(X_test)

print("Data loaded...")

print("With PCA")

# Fit models using PCA
print("Applying PCA...")

for n_c in bar(pca_val):
    # Feature extraction using PCA with number of components as 500
    pca = PCA(n_components=n_c)
    X_train_sd_pca = pca.fit_transform(X_train_sd)
    X_test_sd_pca = pca.transform(X_test_sd)
    pca1 = PCA(n_components=n_c)
    X_train_pca = pca1.fit_transform(X_train)
    X_test_pca = pca1.transform(X_test)
    # Perceptron Model
    pe_pca = Perceptron(n_iter=10, eta0=10, n_jobs=-1)
    pe_pca.fit(X_train_sd_pca, Y_train)

    # Naive Bayes Classification
    nb_pca = GaussianNB()
    nb_pca.fit(X_train_pca, Y_train)

    # Decision Tree Classifier
    dt_pca = DecisionTreeClassifier(max_depth=4)
    dt_pca.fit(X_train_pca, Y_train)

    # RandomForest Classifier
    rf_pca = RandomForestClassifier(criterion='entropy', n_estimators=10, n_jobs=-1)
    rf_pca.fit(X_train_pca, Y_train)

    # Logistic Regression trained with PCA
    lr_pca = LogisticRegression()
    lr_pca.fit(X_train_sd_pca, Y_train)

    # Gaussian Process Classifier with PCA
    gpc_pca = KNeighborsClassifier()
    gpc_pca.fit(X_train_sd_pca, Y_train)

    print("Models with PCA trained")
    print("Testing...")

    # Prediction for Perceptron
    Y_pred_pe_pca = pe_pca.predict(X_test_sd_pca)

    # Prediction for Naive Bayes
    Y_pred_nb_pca = nb_pca.predict(X_test_pca)

    # Prediction for Decision Tree Classifier
    Y_pred_dt_pca = dt_pca.predict(X_test_pca)

    # Prediction for RandomForest Classifier
    Y_pred_rf_pca = rf_pca.predict(X_test_pca)

    # Prediction for Logistic Regression
    Y_pred_lr_pca = lr_pca.predict(X_test_sd_pca)

    # Gaussian Process Classifier
    Y_pred_gpc_pca = gpc_pca.predict(X_test_sd_pca)

    print("PCA n_components : %d" % n_c)

    # Accuracy, precision, recall and F1 score
    for y, x, p in zip([Y_pred_pe_pca, Y_pred_nb_pca, Y_pred_dt_pca, Y_pred_rf_pca, Y_pred_lr_pca, Y_pred_gpc_pca],
                    ['PERCEPTRON PCA', 'NAIVE_BAYES PCA', 'DECISION_TREE PCA',
                     'RANDOM_FOREST PCA', 'LOGISTIC_REGRESSION PCA', 'KNeighborsClassifier'],
                       [pe_v, nb_v, dt_v, rf_v, lr_v, kn_v]):
        print(x)
        error = (Y_test != y).sum()
        p[0].append(error)
        print("Errors    : %d" % error)
        acc = accuracy_score(y, Y_test) * 100
        p[1].append(acc)
        print("Accuracy  : %.2f%%" % acc)
        ps = precision_score(y, Y_test) * 100
        p[2].append(ps)
        print("Precision : %.2f%%" % ps)
        rs = recall_score(y, Y_test) * 100
        p[3].append(rs)
        print("Recall    : %.2f%%" % rs)
        f1 = f1_score(y, Y_test) * 100
        p[4].append(f1)
        print("F1 Score  : %.2f%% \n" % f1)

    print("\n")

print("WITHOUT PCA")
# Perceptron Model
pe = Perceptron(n_iter=10, eta0=10, n_jobs=-1)
pe.fit(X_train_sd, Y_train)

# Naive Bayes Classification
nb = GaussianNB()
nb.fit(X_train, Y_train)

# Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, Y_train)

# RandomForest Classifier
rf = RandomForestClassifier(criterion='entropy', n_estimators=10, n_jobs=-1)
rf.fit(X_train, Y_train)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_sd, Y_train)

# KNeighborsClassifier
gpc = KNeighborsClassifier()
gpc.fit(X_train_sd, Y_train)

print("Models trained")
print("Predicting...")

# Prediction for Perceptron
Y_pred_pe = pe.predict(X_test_sd)

# Prediction for Naive Bayes
Y_pred_nb = nb.predict(X_test)

# Prediction for Decision Tree Classifier
Y_pred_dt = dt.predict(X_test)

# Prediction for RandomForest Classifier
Y_pred_rf = rf.predict(X_test)

# Prediction for Logistic Regression
Y_pred_lr = lr.predict(X_test_sd)

# Prediction for Gaussian Process Classifier
Y_pred_gpc = gpc.predict(X_test_sd)

# Accuracy, precision, recall and F1 score
for y, x, p in zip([Y_pred_pe, Y_pred_nb, Y_pred_dt, Y_pred_rf, Y_pred_lr, Y_pred_gpc],
                ['PERCEPTRON', 'NAIVE_BAYES', 'DECISION_TREE', 'RANDOM_FOREST', 'LOGISTIC_REGRESSION',
                 'KNeighborsClassifier'], [pe_v, nb_v, dt_v, rf_v, lr_v, kn_v]):
    print(x)
    error = (Y_test != y).sum()
    p[0].append(error)
    print("Errors    : %d" % error)
    acc = accuracy_score(y, Y_test) * 100
    p[1].append(acc)
    print("Accuracy  : %.2f%%" % acc)
    ps = precision_score(y, Y_test) * 100
    p[2].append(ps)
    print("Precision : %.2f%%" % ps)
    rs = recall_score(y, Y_test) * 100
    p[3].append(rs)
    print("Recall    : %.2f%%" % rs)
    f1 = f1_score(y, Y_test) * 100
    p[4].append(f1)
    print("F1 Score  : %.2f%% \n" % f1)

print("************************************************************************")
print("\n")
print("Values for graphs")
for i, j in zip(['PERCEPTRON', 'NAIVE_BAYES', 'DECISION_TREE', 'RANDOM_FOREST', 'LOGISTIC_REGRESSION',
                 'KNeighborsClassifier'], [pe_v, nb_v, dt_v, rf_v, lr_v, kn_v]):
    print(i)
    print(j)
    print("**********************************************************************")
    print("\n")
# Plotting Graphs
# x-axis : without PCA,
per_x_axis = [10, 50, 100, 200, 300, 400, 500, 600, 726]
for p, t in zip([pe_v, nb_v, dt_v, rf_v, lr_v, kn_v], ['PERCEPTRON', 'NAIVE_BAYES', 'DECISION_TREE', 'RANDOM_FOREST',
                                                       'LOGISTIC_REGRESSION','KNeighborsClassifier']):

    plt.subplot(1, 2, 1)
    plt.plot(per_x_axis, p[1], marker='o', color='blue', label='Accuracy')
    plt.plot(per_x_axis, p[2], marker='s', color='red', label='Precision')
    plt.plot(per_x_axis, p[3], marker='x', color='green', label='Recall')
    plt.plot(per_x_axis, p[4], marker='D', color='black', label='F1 Score')
    if t != 'NAIVE_BAYES':
        plt.ylim([80.0, 100.0])
    else:
        plt.ylim([10.0, 95.0])
    plt.title('{} Feature Extraction'.format(t))
    plt.xlabel('Features Extracted')
    plt.ylabel('Scores')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(per_x_axis, p[0], label='No. of Errors')
    plt.xlabel('Features Extracted')
    plt.ylabel('Errors')
    plt.title('Error Plot')

    plt.show()
