import os

f = open('preprocess_small_scale.sh', 'w')
for dataset in ['ugr16', 'cidds', 'ton', 'caida', 'dc', 'ca', 'm57']:
    f.write(f'echo {dataset}\n')
    f.write(f'python preprocess_forecasting.py --dataset_name {dataset} --input_file ../../data/conext2023/small-scale/{dataset}/raw_bits.csv --output_folder ../../data/conext2023/small-scale/{dataset}\n')

f.close()