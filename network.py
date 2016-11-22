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
from sklearn.decomposition import PCA

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

# Feature extraction using PCA with number of components as 500
pca = PCA(n_components=500)
X_train_sd_pca = pca.fit_transform(X_train_sd)
X_test_sd_pca = pca.transform(X_test_sd)
pca1 = PCA(n_components=500)
X_train_pca = pca1.fit_transform(X_train)
X_test_pca = pca1.transform(X_test)

print("Data loaded...")

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
for y, x in zip([Y_pred_pe, Y_pred_nb, Y_pred_dt, Y_pred_rf, Y_pred_lr, Y_pred_gpc],
                ['PERCEPTRON', 'NAIVE_BAYES', 'DECISION_TREE', 'RANDOM_FOREST', 'LOGISTIC_REGRESSION',
                 'KNeighborsClassifier']):
    print(x)
    print("Errors    : %d" % (Y_test != y).sum())
    print("Accuracy  : %.2f%%" % (accuracy_score(y, Y_test) * 100))
    print("Precision : %.2f%%" % (precision_score(y, Y_test) * 100))
    print("Recall    : %.2f%%" % (recall_score(y, Y_test) * 100))
    print("F1 Score  : %.2f%% \n" % (f1_score(y, Y_test) * 100))

# Fit models using PCA
print("Applying PCA...")
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

# Accuracy, precision, recall and F1 score
for y, x in zip([Y_pred_pe_pca, Y_pred_nb_pca, Y_pred_dt_pca, Y_pred_rf_pca, Y_pred_lr_pca, Y_pred_gpc_pca],
                ['PERCEPTRON PCA', 'NAIVE_BAYES PCA', 'DECISION_TREE PCA',
                 'RANDOM_FOREST PCA', 'LOGISTIC_REGRESSION PCA', 'KNeighborsClassifier']):
    print(x)
    print("Errors    : %d" % (Y_test != y).sum())
    print("Accuracy  : %.2f%%" % (accuracy_score(y, Y_test) * 100))
    print("Precision : %.2f%%" % (precision_score(y, Y_test) * 100))
    print("Recall    : %.2f%%" % (recall_score(y, Y_test) * 100))
    print("F1 Score  : %.2f%% \n" % (f1_score(y, Y_test) * 100))
