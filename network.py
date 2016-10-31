import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# loading the data set
df = pd.read_csv('LABTESTS_DEATH.csv', header=None)
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

# Perceptron Model
lr = Perceptron(n_iter=10, eta0=10, n_jobs=-1)
lr.fit(X_train_sd, Y_train)

# Naive Bayes Classification
nb = GaussianNB()
nb.fit(X_train, Y_train)

# Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, Y_train)

# RandomForest Classifier
rf = RandomForestClassifier(criterion='entropy', n_estimators=10, n_jobs=-1)
rf.fit(X_train, Y_train)

# Prediction for Logistic Regression
Y_pred = lr.predict(X_test_sd)

# Prediction for Naive Bayes
Y_pred_nb = nb.predict(X_test)

# Prediction for Decision Tree Classifier
Y_pred_dt = dt.predict(X_test)

# Prediction for RandomForest Classifier
Y_pred_rf = rf.predict(X_test)

# Accuracy, precision, recall and F1 score
for y, x in zip([Y_pred, Y_pred_nb, Y_pred_dt, Y_pred_rf],
                ['LOGISTIC_REGRESSION', 'NAIVE_BAYES', 'DECISION_TREE', 'RANDOM_FOREST']):
    print(x)
    print("Errors    : %d" % (Y_test != y).sum())
    print("Accuracy  : %.2f%%" % (accuracy_score(y, Y_test) * 100))
    print("Precision : %.2f%%" % (precision_score(y, Y_test) * 100))
    print("Recall    : %.2f%%" % (recall_score(y, Y_test) * 100))
    print("F1 Score  : %.2f%% \n" % (f1_score(y, Y_test) * 100))
