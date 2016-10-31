import pandas as pd
import numpy as np

# Read the LABTESTS.csv file
df_lab = pd.read_csv('LABTESTS.csv')
# Drop the last column because the SUBJECT_ID 99999 is not present in DEATH.csv
df_lab = df_lab.drop(df_lab.index[[-1]])
# Create a DataFrame with columns as SUBJECT_IDs of the df_lab file with zeros as values
# Zeros as values under each column later helps to map the DEATH values from the DEATH.csv file
df_sub_lab = pd.DataFrame([np.zeros(len(df_lab['SUBJET_ID'].values))], columns=df_lab['SUBJET_ID'].values)

# Read the DEATH.csv file
df_death = pd.read_csv('DEATH.csv')
# Create a DataFrame with columns as SUBJECT_IDs of the df_death file and fill with DEATH value
df_sub_death = pd.DataFrame([df_death['DEATH'].values], columns=df_death['SUBJECT_ID'].values)

# Automatically append the SUBJECT_IDs from the DEATH.csv to LABTESTS.csv file with DEATH value as the 2nd row
df_sub_death = df_sub_lab.append(df_sub_death)

# Drop columns with Nan values in the df_sub_death Dataframe
df_sub_death = df_sub_death.dropna(axis=1)

# Remove the initially added zeros row and convert the values of the column into a numpy array
df_sub_death = df_sub_death.iloc[1].values

# Add the DEATH column to the df_lab Dataframe
df_lab['DEATH'] = df_sub_death

# Remove the zeros column that is formed at as first column in the df_lab DataFrame
df_lab = df_lab.iloc[:, 1:]
# Create a new csv file which contains the DEATH column along with the previous values from the df_lab DataFrame
df_lab.to_csv('LABTESTS_DEATH.csv')
