import os

# List of folders to be created
folders = ['data/positive', 'data/negative', 'data/anchor']

# Loop through the list and create each folder if it does not exist
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f'Folder "{folder}" created.')  # Confirm folder creation
    else:
        print(f'Folder "{folder}" already exists.')  # Notify if folder already exists
