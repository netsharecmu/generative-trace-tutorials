import numpy as np
import pandas as pd

PARENT_DIR = "exps/pcap/ddpm_mlp_best/"
RAWDATASET_PATH = '../../dataset/caida-raw.csv'

np.set_printoptions(suppress=True,
                    formatter={'float_kind':'{:0.5f}'.format})

pd.set_option('display.float_format', lambda x: '%.5f' % x) # suppress scientific notation 

df_raw = pd.read_csv(RAWDATASET_PATH) 

categoricalColumns = []
for name, values in df_raw.dtypes.items():
    if values == 'object': 
        categoricalColumns.append(name)

df_raw = df_raw.drop(columns=categoricalColumns) 
numericColumns = list(df_raw.columns); 

generated_num = np.load(PARENT_DIR +"X_num_train.npy") 
generated_cat = np.load(PARENT_DIR +"X_cat_train.npy") 

df = pd.concat([pd.DataFrame(generated_num, columns = numericColumns), pd.DataFrame(generated_cat, columns = categoricalColumns)], axis=1)
print(df.head(5))

df.to_csv(PARENT_DIR + '/generated.csv', sep='\t')