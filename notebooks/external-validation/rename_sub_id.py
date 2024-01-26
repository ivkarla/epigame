import os
import pandas as pd
from sys import argv
from os import getcwd

main_folder = getcwd()

path = argv[1]
print(path)

# Replace with your actual folder path and metadata file path
folder_path = path + 'preprocessed/'
metadata_file_path = path + 'metadata.xlsx'

# Read metadata from Excel file
metadata_df = pd.read_excel(metadata_file_path)

# Iterate through files in the folder
for filename in os.listdir(folder_path):
    if filename.startswith("sub-") and filename.endswith(".prep"):
        # Extract subject_id from the filename
        subject_id = filename.split('-')[0]+ "-" + filename.split('-')[1]

        # Find corresponding SCC_ID from metadata
        scc_id = metadata_df.loc[metadata_df['SUB_ID'] == subject_id, 'SCC_ID'].values

        if len(scc_id) == 1:
            # If SCC_ID is found, rename the file
            new_filename = f"{scc_id[0]}.prep"
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)

            os.rename(old_file_path, new_file_path)
            print(f"Renamed file: {filename} to {new_filename}")
        else:
            print(f"SCC_ID not found for SUB_ID: {subject_id}")
