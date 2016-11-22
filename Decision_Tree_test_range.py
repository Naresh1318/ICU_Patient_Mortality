from __future__ import print_function
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

# Parameters
top_feat_no = 10
file_name = 'ICU'

print("Loading data...")

# loading the data set
df = pd.read_csv('LABTESTS_DEATH_V2.csv', header=None)
X = df.iloc[1:, 1:-2].values
Y = df.iloc[1:, -1].values
Y = Y.astype(np.float)

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

# Select the best indices columns from the DataFrame,
# last top_feat_no are chosen since the indices are sorted in the ascending order
df = df[indices[-top_feat_no:]]
# Get only the test values from the DataFrame
X2 = df.iloc[1:, :].values
# Copy the target list
Y2 = Y[:]

print("Features Selected!!!")

# Split the data into training and testing sets
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.1, random_state=0)

# Use Decision Tree Classifier to get the graph, which can be used to get the threshold test values
clf = DecisionTreeClassifier(max_depth=4, criterion='')
clf.fit(X_train2, Y_train2)
clf_pred = clf.predict(X_test2)

# Get the accuracy for Decision Tree
print("%.2f%%" % (accuracy_score(clf_pred, Y_test2) * 100))

# Get the test ids from the new DataFrame
test_ids = df.iloc[0, :].values
test_ids_str = []
for i in test_ids:
    test_ids_str.append(str(i))

# Export the graph
with open("{}.dot".format(file_name), 'w') as f:
    f = tree.export_graphviz(clf, out_file=f, feature_names=test_ids_str, class_names=['0', '1'],
                             filled=True, rounded=True, special_characters=True, impurity=True)

# convert the graph from dot to pdf format
os.system('dot -Tpdf %s.dot -o %s.pdf' % (file_name, file_name))
