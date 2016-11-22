from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Parameters
top_feat_no = 10

print("Loading data...")

# loading the data set
df = pd.read_csv('LABTESTS_DEATH_V2.csv', header=None)
X = df.iloc[1:, 1:-2].values
Y = df.iloc[1:, -1].values
Y = Y.astype(np.float)

# Split train test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

print("Data Loaded!!!")

# Use RandomForests to select features
feat_labels = df.columns[1:-2]
forest = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)
forest.fit(X_train, Y_train)
# Get the feature importance from the RandomForestClassifier, has a high value for important features
importance = forest.feature_importances_
# Get the indices of the important features, 1 is added because the indices start at 0
indices = np.argsort(importance) + 1

# To find the top tests which predict ICU death
test_ids = df.iloc[0, :].values
# Get the test ids for the top tests
top_test_ids = test_ids[indices[-top_feat_no:]]
# Arrange it in descending order
top_test_ids = top_test_ids[::-1]

print(top_test_ids)
# Read the D_LABITEMS file to find the test names corresponding to the top_test_ids
df_labitem = pd.read_csv('D_LABITEMS.csv')
# Print the Column Names
print('SI ITEMID %-*s %-*s %-*s %-*s' % (30, 'LABEL', 30, 'FLUID', 30, 'CATEGORY', 30, 'LOINC_CODE'))
# Get the names of the tests and make a table
for c, i in enumerate(top_test_ids):
    # Find the rows which contain the ITEMID as one of the top_test_ids
    test = df_labitem[df_labitem['ITEMID'] == i].values
    print('%-*d) %d %-*s %-*s %-*s %-*s' % (3, c + 1, test[0, 1], 30, test[0, 2], 30, test[0, 3], 30, test[0, 4],
                                            30, test[0, 5]))
