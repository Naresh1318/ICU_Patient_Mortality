import numpy as np
import pandas as pd
import progressbar

# Create a progressbar object with the ETA widget
bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
])

# Read the LABEVENTS.csv file
df = pd.read_csv('LABEVENTS.csv')

# Select the 2nd, 4th and 6th columns which contain the SUBJECT_ID, ITEM_ID and VALUE respectively
X = df.iloc[:, [1, 3, 5]]

# Variables used later
# Create a list for SUBJECT_IDs
Unique_Sub_id = list(X['SUBJECT_ID'])

# Find the unique SUBJECT_IDs from the list
Unique_Sub_id = np.unique(np.array(Unique_Sub_id))

# Number of unique ITEMIDs from the LABEVENTS.csv file
Unique_Item_id = np.unique(np.array(list(X['ITEMID'])))

# Create a DataFrame with the unique ITEMIDs as the columns
final_df = pd.DataFrame(columns=[Unique_Item_id])

# Iterate through all the unique SUBJECT_IDs
for sub in bar(Unique_Sub_id):

    # Create an empty DataFrame with SUBJECT_ID, ITEMID and VALUE as the column names
    temp2_df = pd.DataFrame(columns=['SUBJECT_ID', 'ITEMID', 'VALUE'])

    # Create a Dataframe for each SUBJECT_ID with [ITEMID, VALUE] as its contents
    # Each SUBJECT_ID can have more that one values for each ITEMID
    df_sub = X.loc[X['SUBJECT_ID'] == sub]

    # Get a list of the unique ITEMIDs for each SUBJECT_ID
    sub_unique_id = df_sub['ITEMID'].unique()

    # Iterate through all the unique ITEMIDs for each SUBJECT_ID
    for item in sub_unique_id:
        # Create a Dataframe for each ITEMID with values for each ITEMID as its contents
        # This is created to find the mean later on
        df_sub_item = df_sub.loc[df_sub['ITEMID'] == item]

        # Remove all the non numerical values for each ITEMID
        df_sub_item = df_sub_item[df_sub_item['VALUE'].apply(lambda y: str(y).isdigit())]

        # Convert the datatype of the elements to int class
        df_sub_item = df_sub_item.astype('int')

        # Find the mean for each ITEMID -> VALUE
        avg_sub_item = df_sub_item['VALUE'].mean(axis=0)

        # Create a new DataFrame with appropriate SUBJECT_ID, ITEMID and VALUE values
        temp_df = pd.DataFrame([[sub, item, avg_sub_item]], columns=['SUBJECT_ID', 'ITEMID', 'VALUE'])
        # Append all values for each SUBJECT_ID to temp2_df
        temp2_df = temp2_df.append(temp_df)
    # Create a new DataFrame for each SUBJECT_ID with the columns as ITEMIDs
    # and each column having the appropriate mean value
    temp3_df = pd.DataFrame([list(temp2_df['VALUE'])], columns=list(temp2_df['ITEMID']))

    # Append temp3_df with the final_df DataFrame, this automatically fills the missing values in the final_df
    # columns with Nan values
    final_df = final_df.append(temp3_df)

# Replace the Nan values with zeros
final_df = final_df.fillna(0)

# Add a new column with SUBJECT_ID as the name and unique subjects as the values
# This also ensures that each row in the final_df file matches to the correct SUBJECT_ID
final_df['SUBJET_ID'] = Unique_Sub_id

# Create the csv file
final_df.to_csv('LABTESTS.csv')
