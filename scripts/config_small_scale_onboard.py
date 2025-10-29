'''
A hacky experiment configuration file. Probably a bad idea.
'''

import os
import copy
import toml
import json
import shutil

import pandas as pd

from config_io import Config

# Change this if you are working on different platforms
NETGPT_BASE_FOLDER = '/ocean/projects/cis230086p/yyin4/DeepStore'

# {model_name: [a list of configs]}. The first key is always the model name, for ease of parsing
configs = Config()

# NetShare
# import netshare.configs.default as netshare_default_configs
# print(netshare_default_configs.__path__)
configs['netshare'] = Config()
base_config_netflow = Config.load_from_file(
    os.path.join('src/netshare/examples/netflow/config_example_netflow_nodp.json'),
    default_search_paths=['src/netshare/netshare/configs/default']
)
for dataset in ['ugr16', 'cidds', 'ton']:
    c = copy.deepcopy(base_config_netflow)
    c.global_config.original_data_file = f'../data/small-scale/{dataset}/raw.csv'
    c.pre_post_processor.config.truncate = 'none'
    c.global_config.n_chunks = 1
    c.model.config.batch_size = 512
    if dataset == 'cidds' or dataset == 'ton':
        c.model.config.sample_len = [10]
    else:
        c.model.config.sample_len = [1]
    c.model.config.epochs = 400
    c.model.config.epoch_checkpoint_freq = 50
    if dataset in ['cidds', 'ton']:
        # cidds and ton has additional columns 'label'
        c.pre_post_processor.config.timeseries.append(
            {
                "column": "label",
                "type": "string",
                "encoding": "categorical"
            }
        )
    configs['netshare'][dataset] = c

base_config_pcap = Config.load_from_file(
    os.path.join('src/netshare/examples/pcap/config_example_pcap_nodp.json'),
    default_search_paths=['src/netshare/netshare/configs/default']
)
for dataset in ['caida', 'dc', 'ca', 'm57']:
    c = copy.deepcopy(base_config_pcap)
    c.global_config.original_data_file = f'../data/small-scale/{dataset}/raw.csv'
    c.pre_post_processor.config.truncate = 'none'
    c.pre_post_processor.config.max_flow_len = 5000
    c.global_config.n_chunks = 1
    c.model.config.batch_size = 512
    c.model.config.epochs = 400
    c.model.config.epoch_checkpoint_freq = 50
    c.model.config.sample_len = [10]
    configs['netshare'][dataset] = c


# REaLTabFormer (tabular)
configs['realtabformer-tabular'] = Config()
for dataset in ['caida', 'dc', 'ca', 'ugr16', 'cidds', 'ton', 'm57']:
    configs['realtabformer-tabular'][dataset] = Config(
        {
            "raw_csv_file": f'../data/small-scale/{dataset}/raw.csv',
            "n_layer": 3,
            "n_head": 4,
            "n_embd": 128,
            "logging_steps": 1000,
            "save_steps": 1000,
            "save_total_limit": None,
            "eval_steps": None,
            "epochs": 100,
            "num_bootstrap": 100
        }
    )

# REaLTabFormer (relational)
configs['realtabformer-timeseries'] = Config()
for dataset in ['caida', 'dc', 'ca', 'ugr16', 'cidds', 'ton', 'm57']:
    configs['realtabformer-timeseries'][dataset] = Config(
        {
            "raw_csv_file": f'../data/small-scale/{dataset}/raw.csv',
            "n_layer": 3,
            "n_head": 4,
            "n_embd": 128,
            "logging_steps": 1000,
            "save_steps": 1000,
            "save_total_limit": None,
            "eval_steps": None,
            "epochs": 100,
            "num_bootstrap": 10
        }
    )

# CTGAN
configs['ctgan'] = Config()
for dataset in ['caida', 'dc', 'ca', 'ugr16', 'cidds', 'ton', 'm57']:
    common_discrete_columns = \
        [f"srcip_{31-i}" for i in range(32)] + \
        [f"dstip_{31-i}" for i in range(32)] + \
        [f"srcport_{15-i}" for i in range(16)] + \
        [f"dstport_{15-i}" for i in range(16)] + \
        ['proto']
    if dataset in ['caida', 'dc', 'ca', 'm57']:
        discrete_columns = common_discrete_columns + ['flag']
    if dataset in ['ugr16', 'cidds', 'ton']:
        discrete_columns = common_discrete_columns + ['type']
        if dataset in ['cidds', 'ton']:
            discrete_columns += ['label']
    configs['ctgan'][dataset] = Config(
        {
            "raw_csv_file": f'../data/small-scale/{dataset}/raw_bits.csv',
            "discrete_columns": discrete_columns
        }
    )

# TVAE
configs['tvae'] = Config()
for dataset in ['caida', 'dc', 'ca', 'ugr16', 'cidds', 'ton', 'm57']:
    common_discrete_columns = \
        [f"srcip_{31-i}" for i in range(32)] + \
        [f"dstip_{31-i}" for i in range(32)] + \
        [f"srcport_{15-i}" for i in range(16)] + \
        [f"dstport_{15-i}" for i in range(16)] + \
        ['proto']
    if dataset in ['caida', 'dc', 'ca', 'm57']:
        discrete_columns = common_discrete_columns + ['flag']
    if dataset in ['ugr16', 'cidds', 'ton']:
        discrete_columns = common_discrete_columns + ['type']
        if dataset in ['cidds', 'ton']:
            discrete_columns += ['label']
    configs['tvae'][dataset] = Config(
        {
            "raw_csv_file": f'../data/small-scale/{dataset}/raw_bits.csv',
            "discrete_columns": discrete_columns
        }
    )

# TabDDPM
configs['tabddpm'] = Config()
for dataset in ['caida', 'dc', 'ca', 'ugr16', 'cidds', 'ton', 'm57']:
    common_discrete_columns = \
        [f"srcip_{31-i}" for i in range(32)] + \
        [f"dstip_{31-i}" for i in range(32)] + \
        [f"srcport_{15-i}" for i in range(16)] + \
        [f"dstport_{15-i}" for i in range(16)] + \
        ['proto']
    if dataset in ['caida', 'dc', 'ca', 'm57']:
        discrete_columns = common_discrete_columns + ['flag']
        target_column = 'pkt_len'
    if dataset in ['ugr16', 'cidds', 'ton']:
        target_column = 'byt'
        discrete_columns = common_discrete_columns + ['type']
        if dataset in ['cidds', 'ton']:
            discrete_columns += ['label']

    configs['tabddpm'][dataset] = Config(
        {
            "raw_csv_file": f'../data/small-scale/{dataset}/raw_bits.csv',
            "discrete_columns": discrete_columns,
            "target_column": target_column,
        }
    )

# Crossformer
configs['crossformer'] = Config()
for dataset in ['caida', 'dc', 'ca', 'ugr16', 'cidds', 'ton', 'm57']:
    configs['crossformer'][dataset] = Config(
        {
            "raw_csv_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/raw_bits_transformed_forecasting.csv'),
            "column_info_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/column_info_transformed_forecasting.json'),
            "encoder_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/encoder_transformed_forecasting.pkl'),
        }
    )

# D3VAE
configs['d3vae'] = Config()
for dataset in ['caida', 'dc', 'ca', 'ugr16', 'cidds', 'ton', 'm57']:
    configs['d3vae'][dataset] = Config(
        {
            "raw_csv_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/raw_bits_transformed_forecasting.csv'),
            "column_info_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/column_info_transformed_forecasting.json'),
            "encoder_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/encoder_transformed_forecasting.pkl'),
        }
    )

# SCINET
configs['scinet'] = Config()
for dataset in ['caida', 'dc', 'ca', 'ugr16', 'cidds', 'ton', 'm57']:
    configs['scinet'][dataset] = Config(
        {
            "raw_csv_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/raw_bits_transformed_forecasting.csv'),
            "column_info_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/column_info_transformed_forecasting.json'),
            "encoder_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/encoder_transformed_forecasting.pkl'),
        }
    )

# DLinear
configs['dlinear'] = Config()
for dataset in ['caida', 'dc', 'ca', 'ugr16', 'cidds', 'ton', 'm57']:
    configs['dlinear'][dataset] = Config(
        {
            "raw_csv_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/raw_bits_transformed_forecasting.csv'),
            "column_info_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/column_info_transformed_forecasting.json'),
            "encoder_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/encoder_transformed_forecasting.pkl'),
        }
    )

# PatchTST
configs['patchtst'] = Config()
for dataset in ['caida', 'dc', 'ca', 'ugr16', 'cidds', 'ton', 'm57']:
    configs['patchtst'][dataset] = Config(
        {
            "raw_csv_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/raw_bits_transformed_forecasting.csv'),
            "column_info_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/column_info_transformed_forecasting.json'),
            "encoder_file": os.path.join(NETGPT_BASE_FOLDER, f'data/small-scale/{dataset}/encoder_transformed_forecasting.pkl'),
        }
    )

# # Export configs to disk
# os.makedirs('configs', exist_ok=True)
# for model_name, dataset_configs in configs.items():
#     for dataset, single_config in dataset_configs.items():
#         with open(os.path.join('configs', f'{model_name}_{dataset}.json'), 'w') as f:
#             json.dump(single_config, f, indent=4)