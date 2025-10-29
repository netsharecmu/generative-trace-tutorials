import csv
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
    
RAWDATASET_PATH = '../../dataset/caida-raw.csv'
PROCESSED_DATA_PATH = 'data/pcap/'
DATASET_SIZE = 14240
TARGET_FEATURE = 'pkt_len' #td for netflow
    
df = pd.read_csv(RAWDATASET_PATH)
df = df.truncate(after=DATASET_SIZE)

if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

data_split = [0.7, 0.3, 0]
type_map = {'train':0, 'val':1, 'test':2}

train_num = int(len(df)*data_split[0])
test_num = int(len(df)*data_split[2])
val_num = len(df) - train_num - test_num 

print ("train_num, val_num, test_num: ", train_num, val_num, test_num)

# Select categorical columns
categorical_columns = df.select_dtypes(include=['object'])

# Convert categorical columns to NumPy array
categorical_array = categorical_columns.to_numpy(dtype=str)
print(categorical_array.shape)

# Save the array as a NumPy file
np.save(PROCESSED_DATA_PATH+ 'X_cat_train.npy', categorical_array[:train_num])
np.save(PROCESSED_DATA_PATH + 'X_cat_val.npy', categorical_array[train_num:train_num+val_num])
np.save(PROCESSED_DATA_PATH + 'X_cat_test.npy', categorical_array[train_num+val_num:train_num+val_num+test_num])

# Select categorical columns
numeric_columns = df.select_dtypes(include=['int64'])

# Convert categorical columns to NumPy array
numeric_array = numeric_columns.to_numpy()

# Save the array as a NumPy file
np.save(PROCESSED_DATA_PATH + 'X_num_train.npy', numeric_array[:train_num])
np.save(PROCESSED_DATA_PATH + 'X_num_val.npy', numeric_array[train_num:train_num+val_num])
np.save(PROCESSED_DATA_PATH + 'X_num_test.npy', numeric_array[train_num+val_num:train_num+val_num+test_num])

y_col = df[TARGET_FEATURE]
np.save(PROCESSED_DATA_PATH + 'y_train.npy', y_col[:train_num])
np.save(PROCESSED_DATA_PATH + 'y_val.npy', y_col[train_num:train_num+val_num])
np.save(PROCESSED_DATA_PATH + 'y_test.npy', y_col[train_num+val_num:train_num+val_num+test_num])

data = {
    "task_type": "regression",
    "name": "Pcap",
    "id": "pcap-id",
    "train_size": train_num,
    "val_size": val_num,
    "test_size": test_num,
    "n_num_features": len(numeric_columns.columns), # 14
    "n_cat_features": len(categorical_columns.columns) #1
}

with open(PROCESSED_DATA_PATH+'info.json', 'w') as f:
    json.dump(data, f, indent = 4)