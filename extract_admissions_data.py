import pandas as pd
import numpy as np

# Read the csv file
df = pd.read_csv('/home/naresh/PycharmProjects/ICU Death/ADMISSIONS.csv')

# Get the values from the 2nd and the 6th column corresponding to SUBJECT_ID and DEATHTIME
X = df.iloc[:, [1, 5]].copy()

# Sorting the dataframe values according to the SUBJECT_IDs
X = X.sort(['SUBJECT_ID', 'DEATHTIME'], ascending=[True, False])

# Convert the Deathtime to Alphabets
death_list = []
sub_list = []
for i in range(len(X)):
    death_time = X.iloc[i, 1]

    # Check if the SUBJECT_IDs repeat, if they do then skip that iteration and continue with the next value
    # If SUBJECTS_IDs do not repeat then append it to sub_list
    try:
        if X.iloc[i, 0] == X.iloc[i + 1, 0]:
            continue
        else:
            sub_list.append(X.iloc[i, 0])
    except:
        continue

    # Nan value in DEATHTIME column corresponds to label ALIVE given by value 1
    # Value 0 corresponds to DEATH
    if death_time is np.nan:
        death_list.append(1)
    else:
        death_list.append(0)

# Create a DataFrame using both the sub_list and death_list
df_death = pd.DataFrame((np.array([sub_list, death_list])).T, columns=['SUBJECT_ID', 'DEATH'])

# Save df_death as a csv file
df_death.to_csv('DEATH.csv')
