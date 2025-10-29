"""
Driver file for calling different generative models

Example usage:

python3 driver.py \
    --config_partition small-scale \
    --dataset_name ugr16 \
    --model_name netshare \
    --cur_time 0
"""

import os
import gc
import json
import time
import copy
import toml
import torch
import joblib
import datetime
import argparse
import subprocess

import pandas as pd
import numpy as np

from config_io import Config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from GPUtil import showUtilization as gpu_usage
# from config import configs # run once to ensure the latest configs are loaded
# print("Finish loading configs")
# from config_small_scale import NETGPT_BASE_FOLDER

def exec_cmd(cmd, wait=True):
    p = subprocess.Popen(cmd, shell=True)
    if wait:
        p.wait()
    returnCode = p.poll()
    if returnCode != 0:
        raise Exception(f"Command {cmd} failed with error code {returnCode}")
    return returnCode

def current_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d%H%M%S") + str(now.microsecond // 1000).zfill(3) # milliseconds precision

def remove_constant_columns(df):
    constant_columns_mapping = {}
    for col in df.columns:
        if len(set(df[col])) == 1:
            constant_columns_mapping[col] = list(set(df[col]))[0]
    df = df.drop(columns=list(constant_columns_mapping.keys()))

    return df, constant_columns_mapping

def main(args):
    # Global timestamp
    if args.cur_time is None:
        cur_time = current_timestamp() # use provided current timestamp from the job script
    else:
        cur_time = args.cur_time # if not provided, just use the current timestamp when running this script
    
    # Load configs
    if args.config_partition == 'small-scale':
        from config_small_scale import configs
        from config_small_scale import NETGPT_BASE_FOLDER
    elif args.config_partition == 'small-scale-onboard':
        from config_small_scale_onboard import configs
        from config_small_scale_onboard import NETGPT_BASE_FOLDER
        args.config_partition = 'small-scale'
    elif args.config_partition == 'small-scale-1':
        from config_small_scale_1 import configs
        from config_small_scale_1 import NETGPT_BASE_FOLDER
        args.config_partition = 'small-scale'
    elif args.config_partition == 'small-scale-2':
        from config_small_scale_2 import configs
        from config_small_scale_2 import NETGPT_BASE_FOLDER
        args.config_partition = 'small-scale'
    elif args.config_partition == 'small-scale-3':
        from config_small_scale_3 import configs
        from config_small_scale_3 import NETGPT_BASE_FOLDER
        args.config_partition = 'small-scale'
    elif args.config_partition == 'small-scale-4':
        from config_small_scale_4 import configs
        from config_small_scale_4 import NETGPT_BASE_FOLDER
        args.config_partition = 'small-scale'
    elif args.config_partition == 'small-scale-5':
        from config_small_scale_5 import configs
        from config_small_scale_5 import NETGPT_BASE_FOLDER
        args.config_partition = 'small-scale'
    elif args.config_partition == 'small-scale-6':
        from config_small_scale_6 import configs
        from config_small_scale_6 import NETGPT_BASE_FOLDER
        args.config_partition = 'small-scale'
    elif args.config_partition == 'pretrain':
        from config_pretrain import configs
    else:
        raise ValueError(f"Unknown config partition: {args.config_partition}")
    
    RESULT_PATH_BASE=os.path.join(NETGPT_BASE_FOLDER, "results")
    RESULT_PATH_BASE_SMALL_SCALE=os.path.join(RESULT_PATH_BASE, "small-scale")
    RESULT_PATH_BASE_MEDIUM_SCALE=os.path.join(RESULT_PATH_BASE, "medium-scale")
    RESULT_PATH_PRETRAIN=os.path.join(RESULT_PATH_BASE, "pretrain")
    RESULT_PATH={
        'small-scale': {
            'runs': os.path.join(RESULT_PATH_BASE_SMALL_SCALE, "runs"), # models/logs/intermediate files etc.
            'csv': os.path.join(RESULT_PATH_BASE_SMALL_SCALE, 'csv'), # raw/synthetic csv files
            'txt': os.path.join(RESULT_PATH_BASE_SMALL_SCALE, 'txt'), # raw/synthetic txt files converted from csv (used for BERT inference)
            'npz': os.path.join(RESULT_PATH_BASE_SMALL_SCALE, 'npz'), # raw/synthetic npz files containing embeddings (and mu, sigma)
            'time': os.path.join(RESULT_PATH_BASE_SMALL_SCALE, 'time'), # time elapsed for each model
        },
        'medium-scale': {
            'runs': os.path.join(RESULT_PATH_BASE_MEDIUM_SCALE, "runs"), # models/logs/intermediate files etc.
            'csv': os.path.join(RESULT_PATH_BASE_MEDIUM_SCALE, 'csv'), # raw/synthetic csv files
            'txt': os.path.join(RESULT_PATH_BASE_MEDIUM_SCALE, 'txt'), # raw/synthetic txt files converted from csv (used for BERT inference)
            'npz': os.path.join(RESULT_PATH_BASE_MEDIUM_SCALE, 'npz'), # raw/synthetic npz files containing embeddings (and mu, sigma)
            'time': os.path.join(RESULT_PATH_BASE_MEDIUM_SCALE, 'time'), # time elapsed for each model
        },
        'pretrain': {
            'runs': os.path.join(RESULT_PATH_PRETRAIN, "runs"), # models/logs/intermediate files etc.
            'csv': os.path.join(RESULT_PATH_PRETRAIN, 'csv'), # raw/synthetic csv files
            'txt': os.path.join(RESULT_PATH_PRETRAIN, 'txt'), # raw/synthetic txt files converted from csv (used for BERT inference)
            'npz': os.path.join(RESULT_PATH_PRETRAIN, 'npz'), # raw/synthetic npz files containing embeddings (and mu, sigma)
            'time': os.path.join(RESULT_PATH_PRETRAIN, 'time'), # time elapsed for each model
        }
    }

    for section, section_path in RESULT_PATH.items():
        for sub_section, sub_section_path in section_path.items():
            os.makedirs(sub_section_path, exist_ok=True)

    model_name, dataset_name = args.model_name, args.dataset_name
    if dataset_name in ['ugr16', 'cidds', 'ton', 'ugr16-100k', 'ugr16-200k', 'ugr16-500k', 'ugr16-800k', 'ugr16-1m', 'netflow-3M-mixed']:
        dataset_type = 'netflow'
    elif dataset_name in ['caida', 'dc', 'ca', 'm57', 'm57-100k', 'm57-200k', 'm57-500k', 'm57-800k', 'm57-1m', 'pcap-3M-mixed'] + [f'm57-{num}' for num in [1200000, 1500000, 1800000, 2000000, 2500000, 3000000, 4500000, 5000000]]:
        dataset_type = 'pcap'
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    work_folder = os.path.join(
        RESULT_PATH[args.config_partition]['runs'],
        f'{model_name}_{dataset_name}_{cur_time}')
    os.makedirs(work_folder, exist_ok=True)
    
    # save configs to work folder
    current_config = Config(configs[model_name][dataset_name])
    current_config.dump_to_file(os.path.join(work_folder, "config_driver.json"))
    current_config_file_path = os.path.join(work_folder, "config_driver.json")

    #===========================================================================
    #================================Run models=================================
    #===========================================================================
    # READ RAW DATA FILE and select related columns (e.g., drop `version`/`ihl`/`chksum`)
    if 'raw_csv_file' in current_config: # realtabformer-tabular, realtabformer-timeseries, ctgan, tvae, tabddpm, crossformer, d3vae, scinet, dlinear, patchtst
        df = pd.read_csv(current_config.raw_csv_file)
        if dataset_type == "pcap":
            dropped_columns = []
            for col in ['version', 'ihl', 'chksum']:
                if col in df.columns:
                    dropped_columns.append(col)
            df.drop(columns=dropped_columns, inplace=True)
        print(df.shape)
        print(df.columns)

    start_time = time.time()
    end_time_train = None
    if model_name == "netshare":
        import netshare.ray as ray
        from netshare import Generator

        ray.config.enabled = False
        ray.init(address="auto")
        generator = Generator(config=current_config_file_path)
        generator.train_and_generate(work_folder=work_folder)
        syn_df = pd.read_csv(generator._pre_post_processor.best_syndf_filename_list[0])
        ray.shutdown()
    

    if model_name == "realtabformer-tabular":
        from realtabformer import REaLTabFormer
        from transformers.models.gpt2 import GPT2Config

        # Non-relational or parent table.
        rtf_model = REaLTabFormer(
            model_type="tabular",
            tabular_config=GPT2Config(
                n_layer=getattr(current_config, 'n_layer', 12),
                n_head=getattr(current_config, 'n_head', 12),
                n_embd=getattr(current_config, 'n_embd', 768)
            ),
            checkpoints_dir=os.path.join(work_folder, "rtf_checkpoints"),
            samples_save_dir=os.path.join(work_folder, "rtf_samples"),
            gradient_accumulation_steps=4,
            epochs=current_config.epochs,
            batch_size=16,
            logging_steps=current_config.logging_steps,
            save_steps=current_config.save_steps,
            save_total_limit=current_config.save_total_limit,
            eval_steps=current_config.eval_steps)

        rtf_model.fit(df, num_bootstrap=current_config.num_bootstrap)
        rtf_model.save(os.path.join(work_folder, "rtf_model"))
        syn_df = rtf_model.sample(n_samples=len(df), gen_batch=1024)

    
    if model_name == "realtabformer-timeseries":
        from realtabformer import REaLTabFormer
        from pathlib import Path

        from transformers import EncoderDecoderConfig
        from transformers.models.gpt2 import GPT2Config

        join_on = "flow_id"
        session_keys = ["srcip", "dstip", "srcport", "dstport", "proto"]
        df[join_on] = df.groupby(session_keys).ngroup()
        timeseries_keys = list(set(df.columns) - set(session_keys + [join_on]))
        print("Raw df:", df.columns, df.shape)

        # Metadata (parent table)
        parent_df = pd.DataFrame(df.groupby(list(session_keys+[join_on])).groups.keys(), columns=list(session_keys+[join_on]))
        print("parent_df:", parent_df.columns, parent_df.shape)

        # Timeseries (child table)
        child_df = df[list(timeseries_keys+[join_on])]
        print("child_df:", child_df.columns, child_df.shape)

        # Make sure that the key columns in both the
        # parent and the child table have the same name.
        assert ((join_on in parent_df.columns) and
                (join_on in child_df.columns))

        # Non-relational or parent table. Don't include the
        # unique_id field.
        parent_model = REaLTabFormer(
            model_type="tabular",
            tabular_config=GPT2Config(
                n_layer=getattr(current_config, 'n_layer', 12),
                n_head=getattr(current_config, 'n_head', 12),
                n_embd=getattr(current_config, 'n_embd', 768)
            ),
            checkpoints_dir=os.path.join(work_folder, "rtf_checkpoints"),
            samples_save_dir=os.path.join(work_folder, "rtf_samples"),
            gradient_accumulation_steps=4,
            epochs=current_config.epochs,
            batch_size=64,
            logging_steps=current_config.logging_steps,
            save_steps=current_config.save_steps,
            save_total_limit=current_config.save_total_limit,
            eval_steps=current_config.eval_steps)
        parent_model.fit(
            parent_df.drop(join_on, axis=1), 
            num_bootstrap=current_config.num_bootstrap)

        pdir = Path(os.path.join(work_folder, "rtf_parent/"))
        parent_model.save(pdir)

        # # Get the most recently saved parent model,
        # # or a specify some other saved model.
        # parent_model_path = pdir / "idXXX"
        parent_model_path = sorted([
            p for p in pdir.glob("id*") if p.is_dir()],
            key=os.path.getmtime)[-1]

        child_model = REaLTabFormer(
            model_type="relational",
            relational_config=EncoderDecoderConfig(
                encoder=GPT2Config(
                    n_layer=getattr(current_config, 'n_layer', 12),
                    n_head=getattr(current_config, 'n_head', 12),
                    n_embd=getattr(current_config, 'n_embd', 768)
                ).to_dict(),
                decoder=GPT2Config(
                    n_layer=getattr(current_config, 'n_layer', 12),
                    n_head=getattr(current_config, 'n_head', 12),
                    n_embd=getattr(current_config, 'n_embd', 768)
                ).to_dict(),
            ),
            parent_realtabformer_path=parent_model_path, # set to `None` if experiencing the issue when fitting the child model: https://github.com/worldbank/REaLTabFormer/issues/22
            checkpoints_dir=os.path.join(work_folder, "rtf_checkpoints"),
            samples_save_dir=os.path.join(work_folder, "rtf_samples"),
            output_max_length=512,
            train_size=0.8,
            gradient_accumulation_steps=4,
            epochs=current_config.epochs,
            batch_size=16,
            logging_steps=current_config.logging_steps,
            save_steps=current_config.save_steps,
            save_total_limit=current_config.save_total_limit,
            eval_steps=current_config.eval_steps)

        child_model.fit(
            df=child_df,
            in_df=parent_df,
            join_on=join_on,
            num_bootstrap=current_config.num_bootstrap)
        
        pdir = Path(os.path.join(work_folder, "rtf_child/"))
        child_model.save(pdir)

        # Generate parent samples.
        print("Start generating parent samples...")
        parent_samples = parent_model.sample(len(parent_df), gen_batch=1024)
        print("Finish generating parent samples...")

        # Create the unique ids based on the index.
        parent_samples.index.name = join_on
        parent_samples = parent_samples.reset_index()

        # Generate the relational observations.
        print("Start generating child samples...")
        child_samples = child_model.sample(
            input_unique_ids=parent_samples[join_on],
            input_df=parent_samples.drop(join_on, axis=1),
            gen_batch=64)
        child_samples.index.name = join_on
        print("Finish generating child samples...")

        print("Parent df:", parent_samples.shape, parent_samples.columns)
        print(parent_samples.head())
        print("Child df:", child_samples.shape, child_samples.columns)
        print(child_samples.head())

        syn_df = child_samples.merge(parent_samples, on=join_on) # merge parent and child tables
        assert syn_df.shape[0] == child_samples.shape[0] # sanity check

        # Child table is huge... cleaning up GPU memory for BERT inference later
        print("Initial GPU Usage")
        gpu_usage() 
        del parent_model, child_model
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU Usage after emptying the cache")
        gpu_usage()

    if model_name == "ctgan":
        from ctgan import CTGAN

        discrete_columns = current_config['discrete_columns']
        print("discrete_columns:", discrete_columns)

        ctgan = CTGAN(epochs=100, verbose=True)
        print("Start CTGAN training...")
        ctgan.fit(df, discrete_columns)
        print("CTGAN training finished...")
        ctgan.save(os.path.join(work_folder, "model.pt"))
        print("CTGAN model saved...")
        syn_df = ctgan.sample(len(df))
        print("CTGAN sampling finished...")
    

    if model_name == "tvae":
        from ctgan import TVAE
        
        discrete_columns = current_config['discrete_columns']
        print("discrete_columns:", discrete_columns)

        tvae = TVAE(epochs=100) # verbose=True is not supported for now
        print("Start TVAE training...")
        tvae.fit(df, discrete_columns)
        print("TAVE training finished...")
        tvae.save(os.path.join(work_folder, "model.pt"))
        print("TAVE model saved...")
        syn_df = tvae.sample(len(df))
        print("TAVE sampling finished...")

    
    if model_name == "tabddpm":
        data_split = [0.7, 0.3, 0] # train, val, test; no test in our case
        type_map = {'train':0, 'val':1, 'test':2}

        # remove constant columns
        # df, constant_columns_mapping = remove_constant_columns(df)
        # for constant_col, val in constant_columns_mapping.items():
        #     if constant_col in current_config.discrete_columns:
        #         current_config.discrete_columns.remove(constant_col)

        def split_dataframe(input_df, ratios, seed=42):
            df = input_df.copy(deep=True)
            assert sum(ratios) == 1.0, "Sum of ratios must be 1.0"

            # Compute the split sizes
            train_ratio, val_ratio, test_ratio = ratios

            if test_ratio == 0:
                # Split the data into train and validation sets
                train, val = train_test_split(df, test_size=val_ratio, random_state=seed)
                return list(train.index), list(val.index), []
            else:
                # First, split the data into train+val and test
                train_val, test = train_test_split(df, test_size=test_ratio, random_state=seed)

                # Compute the new ratio for the validation set (original ratio relative to train+val)
                relative_val_ratio = val_ratio / (1 - test_ratio)

                # Split the train+val set into train and validation sets
                train, val = train_test_split(train_val, test_size=relative_val_ratio, random_state=seed)

                return list(train.index), list(val.index), list(test.index)
    

        # train/val/test
        train_idx, val_idx, test_idx = split_dataframe(df, data_split)
        print ("train_num, val_num, test_num: ", len(train_idx), len(val_idx), len(test_idx))

        os.makedirs(os.path.join(work_folder, "raw_dataset"), exist_ok=True) # preprocessed raw dataset

        # Categorical
        categorical_columns = copy.deepcopy(current_config.discrete_columns)
        if current_config.target_column in categorical_columns:
            categorical_columns.remove(current_config.target_column)
        print("Categorical columns: ", categorical_columns)
        categorical_array = df[categorical_columns].to_numpy(dtype=str)
        print("Catetgorical array shape:", categorical_array.shape)
        np.save(os.path.join(work_folder, "raw_dataset", "X_cat_train.npy"), categorical_array[train_idx])
        np.save(os.path.join(work_folder, "raw_dataset", "X_cat_val.npy"), categorical_array[val_idx])
        np.save(os.path.join(work_folder, "raw_dataset", "X_cat_test.npy"), categorical_array[test_idx])

        # Continuous
        numeric_columns = sorted(list(set(df.columns) - set(current_config.discrete_columns)))
        if current_config.target_column in numeric_columns:
            numeric_columns.remove(current_config.target_column)
        print("Numeric columns:", numeric_columns)
        numeric_array = df[numeric_columns].to_numpy()
        print("Numeric array shape: ", numeric_array.shape)
        np.save(os.path.join(work_folder, "raw_dataset", "X_num_train.npy"), numeric_array[train_idx])
        np.save(os.path.join(work_folder, "raw_dataset", "X_num_val.npy"), numeric_array[val_idx])
        np.save(os.path.join(work_folder, "raw_dataset", "X_num_test.npy"), numeric_array[test_idx])

        # y_col only used for regression task on val set (for picking the best model of generation)
        y_col = df[current_config.target_column]
        np.save(os.path.join(work_folder, "raw_dataset", "y_train.npy"), y_col[train_idx])
        np.save(os.path.join(work_folder, "raw_dataset", "y_val.npy"), y_col[val_idx])
        np.save(os.path.join(work_folder, "raw_dataset", "y_test.npy"), y_col[test_idx])

        # Config native to tabddpm
        config_info = {
            "task_type": "regression",
            "name": f"{model_name}_{dataset_name}",
            "id": f"{cur_time}",
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "n_num_features": len(numeric_columns),
            "n_cat_features": len(categorical_columns),
        }

        # Export info json
        with open(os.path.join(work_folder, "raw_dataset", "info.json"), 'w') as f:
            json.dump(config_info, f, indent=4)
        
        # Read basic toml
        basic_toml_config = toml.load('src/tabddpm/base-config.toml')
        os.makedirs(os.path.join(work_folder, "results"), exist_ok=True) # results folder
        toml_config = copy.deepcopy(basic_toml_config)
        toml_config['parent_dir'] = os.path.abspath(os.path.join(work_folder, "results"))
        toml_config['real_data_path'] = os.path.abspath(os.path.join(work_folder, "raw_dataset"))
        toml_config['num_numerical_features'] = len(numeric_columns)
        toml_config['device'] = "cuda:0"
        toml_config['sample']['num_samples'] = len(df)

        # Customize config for each dataset as TabDDPM is very hard to train
        if dataset_name == 'caida':
            toml_config['diffusion_params']['num_timesteps'] = 1000
            toml_config['model_params']['rtdl_params']['d_layers'] = [128, 128]
            toml_config['train']['main']['batch_size'] = 1024
            toml_config['train']['main']['steps'] = 1000
            toml_config['train']['main']['lr'] = 0.001
        elif dataset_name == 'dc':
            toml_config['diffusion_params']['num_timesteps'] = 1000
            toml_config['model_params']['rtdl_params']['d_layers'] = [128, 128]
            toml_config['train']['main']['batch_size'] = 1024
            toml_config['train']['main']['steps'] = 1000
            toml_config['train']['main']['lr'] = 0.001
        elif dataset_name == 'ca':
            toml_config['diffusion_params']['num_timesteps'] = 1000
            toml_config['model_params']['rtdl_params']['d_layers'] = [128, 128]
            toml_config['train']['main']['batch_size'] = 1024
            toml_config['train']['main']['steps'] = 1000
            toml_config['train']['main']['lr'] = 0.001
        elif dataset_name == 'm57':
            toml_config['diffusion_params']['num_timesteps'] = 1000
            toml_config['model_params']['rtdl_params']['d_layers'] = [128, 128]
            toml_config['train']['main']['batch_size'] = 1024
            toml_config['train']['main']['steps'] = 1000
            toml_config['train']['main']['lr'] = 0.001
        elif dataset_name == 'ugr16':
            toml_config['diffusion_params']['num_timesteps'] = 1000
            toml_config['model_params']['rtdl_params']['d_layers'] = [128, 128]
            toml_config['train']['main']['batch_size'] = 512
            toml_config['train']['main']['steps'] = 1500
            toml_config['train']['main']['lr'] = 0.001
        elif dataset_name == 'cidds':
            toml_config['diffusion_params']['num_timesteps'] = 1000
            toml_config['model_params']['rtdl_params']['d_layers'] = [128, 128]
            toml_config['train']['main']['batch_size'] = 1024
            toml_config['train']['main']['steps'] = 1000
            toml_config['train']['main']['lr'] = 0.001
        elif dataset_name == 'ton':
            toml_config['diffusion_params']['num_timesteps'] = 1000
            toml_config['model_params']['rtdl_params']['d_layers'] = [128, 128]
            toml_config['train']['main']['batch_size'] = 1024
            toml_config['train']['main']['steps'] = 1000
            toml_config['train']['main']['lr'] = 0.001
        else:
            raise ValueError(f"TabDDPM: Unknown dataset name: {dataset_name}")
        print(toml_config)
        with open(os.path.join(work_folder, 'config.toml'), 'w') as f:
            toml.dump(toml_config, f)
        
        # Run TabDDPM (use conda env `tabddpm`)
        exec_cmd(
            f"cd src/tabddpm && python3 scripts/pipeline.py \
                --config {os.path.abspath(os.path.join(work_folder, 'config.toml'))} \
                --sample \
                --train"
        )

        syn_numerical = np.load(os.path.join(toml_config['parent_dir'], 'X_num_train.npy'))
        print(syn_numerical[:2, :])
        print("Numerical columns", numeric_columns)
        syn_categorical = np.load(os.path.join(toml_config['parent_dir'], 'X_cat_train.npy'))
        print("Synthetic numerical array:", syn_numerical.shape)
        print("Synthetic categorical array:", syn_categorical.shape)
        syn_target = np.load(os.path.join(toml_config['parent_dir'], 'y_train.npy'))

        syn_df = pd.concat(
            [pd.DataFrame(syn_numerical, columns=numeric_columns), 
             pd.DataFrame(syn_categorical, columns=current_config.discrete_columns),
             pd.DataFrame(syn_target, columns=[current_config.target_column])], 
            axis=1)
        print("Synthetic df:", syn_df.shape, syn_df.columns)
        print(syn_df.head())
    
    elif model_name == "crossformer":
        # train
        exec_cmd(
            f"cd src/crossformer && python3 main_crossformer.py \
                --data {args.dataset_name} \
                --root_path {os.path.split(current_config.raw_csv_file)[0]} \
                --data_path {os.path.split(current_config.raw_csv_file)[1]} \
                --work_folder {work_folder} \
                --checkpoints {os.path.join(work_folder, 'checkpoints')} \
                --data_dim {df.shape[1]-1} \
                --in_len 168 \
                --out_len 24 \
                --seg_len 6 \
                --data_split '0.7,0.3,0.0' \
                --skip_testing True \
                "
        )

        end_time_train = time.time()

        # generation
        exec_cmd(
            f"cd src/crossformer && python3 generate_crossformer.py \
                --setting_name Crossformer_{args.dataset_name}_il168_ol24_sl6_win2_fa10_dm256_nh4_el3_itr0 \
                 --work_folder {work_folder} \
                --checkpoint_root {os.path.join(work_folder, 'checkpoints')} \
                --save_pred \
                --inverse \
                --batch_size 1 \
                --sample_size {df.shape[0]} \
                "
        )

        syn_npy = np.load(os.path.join(work_folder, 'results', f'Crossformer_{dataset_name}_il168_ol24_sl6_win2_fa10_dm256_nh4_el3_itr0', 'pred.npy'))[0]
    
    elif model_name == "d3vae":
        # train
        exec_cmd(
            f"cd src/d3vae && python3 main.py \
                --data {args.dataset_name} \
                --root_path {os.path.split(current_config.raw_csv_file)[0]} \
                --data_path {os.path.split(current_config.raw_csv_file)[1]} \
                --work_folder {work_folder} \
                --checkpoints {os.path.join(work_folder, 'checkpoints')} \
                --freq 's' \
                --input_dim {df.shape[1]-1} \
                --target_dim {df.shape[1]-1} \
                --percentage 1.0 \
                --diff_steps 1000 \
                --features='M' \
                --beta_end 0.1 \
                --inverse True \
                --itr 1 \
                --skip_testing True \
                --data_split '0.7,0.3,0.0' \
                "
        )

        end_time_train = time.time()

        # generation
        exec_cmd(
            f"cd src/d3vae && python3 main.py \
                --data {args.dataset_name} \
                --root_path {os.path.split(current_config.raw_csv_file)[0]} \
                --data_path {os.path.split(current_config.raw_csv_file)[1]} \
                --work_folder {work_folder} \
                --checkpoints {os.path.join(work_folder, 'checkpoints')} \
                --freq 's' \
                --input_dim {df.shape[1]-1} \
                --target_dim {df.shape[1]-1} \
                --percentage 1.0 \
                --diff_steps 1000 \
                --features='M' \
                --beta_end 0.1 \
                --inverse True \
                --generate True \
                --itr 1 \
                --skip_testing True \
                --data_split '0.7,0.3,0.0' \
                --sample_size {df.shape[0]} \
                "
        )

        syn_npy = np.load(os.path.join(work_folder, 'results', f'D3VAE_{dataset_name}_sl_16_pl16_0_dim-1_scale0.1_diffsteps1000_itr0', 'pred.npy'))[0]

    elif model_name == "scinet":
        # train
        exec_cmd(
            f"cd src/scinet && python run_network.py \
                --data {args.dataset_name} \
                --root_path {os.path.split(current_config.raw_csv_file)[0]} \
                --data_path {os.path.split(current_config.raw_csv_file)[1]} \
                --work_folder {work_folder} \
                --checkpoints {os.path.join(work_folder, 'checkpoints')} \
                --features M  \
                --seq_len 48 \
                --label_len 24 \
                --pred_len 24 \
                --hidden-size 4 \
                --stacks 1 \
                --levels 3 \
                --lr 3e-3 \
                --batch_size 64 \
                --train_epochs 20 \
                --dropout 0.5 \
                --inverse True \
                --skip_testing True \
                --data_split '0.7,0.3,0.0' \
                --freq 's' \
                "
        )

        end_time_train = time.time()

        # generation
        exec_cmd(
            f"cd src/scinet && python run_network.py \
                --data {args.dataset_name} \
                --root_path {os.path.split(current_config.raw_csv_file)[0]} \
                --data_path {os.path.split(current_config.raw_csv_file)[1]} \
                --work_folder {work_folder} \
                --checkpoints {os.path.join(work_folder, 'checkpoints')} \
                --features M  \
                --seq_len 48 \
                --label_len 24 \
                --pred_len 24 \
                --hidden-size 4 \
                --stacks 1 \
                --levels 3 \
                --lr 3e-3 \
                --batch_size 64 \
                --train_epochs 20 \
                --dropout 0.5 \
                --generate True \
                --inverse True \
                --save True \
                --sample_size {df.shape[0]} \
                --skip_testing True \
                --data_split '0.7,0.3,0.0' \
                --freq 's' \
                "
        )

        syn_npy = np.load(
            os.path.join(
            work_folder, 
            'results', 
            f'SCINet_{dataset_name}_ftM_sl48_ll24_pl24_lr0.003_bs64_hid4.0_s1_l3_dp0.5_invTrue_itr0', 
            'pred_scales.npy'))[0]

    if model_name == "dlinear":
        if dataset_type == "netflow":
            target_col = "byt"
        elif dataset_type == "pcap":
            target_col = "pkt_len"
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # train
        exec_cmd(
            f"cd src/patchtst_supervised && python3 run_longExp.py \
                --is_training 1 \
                --data custom \
                --freq s \
                --root_path {os.path.split(current_config.raw_csv_file)[0]} \
                --data_path {os.path.split(current_config.raw_csv_file)[1]} \
                --work_folder {work_folder} \
                --checkpoints {os.path.join(work_folder, 'checkpoints')} \
                --model_id DLinear_336_96 \
                --model DLinear \
                --features M \
                --seq_len 336 \
                --pred_len 96 \
                --enc_in {df.shape[1]-1} \
                --des 'Exp' \
                --data_split '0.7,0.3,0.0' \
                --skip_testing True \
                --target {target_col} \
                --itr 1 \
                --batch_size 32 \
                --learning_rate 0.005 \
                "
        )

        end_time_train = time.time()

        exec_cmd(
            f"cd src/patchtst_supervised && python3 run_longExp.py \
                --is_training 0 \
                --sample_size {df.shape[0]} \
                --data custom \
                --freq s \
                --root_path {os.path.split(current_config.raw_csv_file)[0]} \
                --data_path {os.path.split(current_config.raw_csv_file)[1]} \
                --work_folder {work_folder} \
                --checkpoints {os.path.join(work_folder, 'checkpoints')} \
                --model_id DLinear_336_96 \
                --model DLinear \
                --features M \
                --seq_len 336 \
                --pred_len 96 \
                --enc_in {df.shape[1]-1} \
                --des 'Exp' \
                --data_split '0.7,0.3,0.0' \
                --skip_testing True \
                --target {target_col} \
                --itr 1 \
                --batch_size 32 \
                --learning_rate 0.005 \
                "
        )

        syn_npy = np.load(os.path.join(
            work_folder, 
            'results', 
            'DLinear_336_96_DLinear_custom_ftM_sl336_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0', 
            'pred.npy'))[0]
    
    # Use config from `electricity.sh`
    elif model_name == "patchtst":
        if dataset_type == "netflow":
            target_col = "byt"
        elif dataset_type == "pcap":
            target_col = "pkt_len"
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # train
        exec_cmd(
            f"cd src/patchtst_supervised && python3 run_longExp.py \
                --is_training 1 \
                --data custom \
                --freq s \
                --root_path {os.path.split(current_config.raw_csv_file)[0]} \
                --data_path {os.path.split(current_config.raw_csv_file)[1]} \
                --work_folder {work_folder} \
                --checkpoints {os.path.join(work_folder, 'checkpoints')} \
                --model_id PatchTST_336_96 \
                --model PatchTST \
                --features M \
                --seq_len 336 \
                --pred_len 96 \
                --enc_in {df.shape[1]-1} \
                --des 'Exp' \
                --data_split '0.7,0.3,0.0' \
                --skip_testing True \
                --target {target_col} \
                --itr 1 \
                --batch_size 32 \
                --learning_rate 0.0001 \
                --e_layers 3 \
                --n_heads 16 \
                --d_model 128 \
                --d_ff 256 \
                --dropout 0.2 \
                --fc_dropout 0.2 \
                --head_dropout 0 \
                --patch_len 16 \
                --stride 8 \
                --patience 10 \
                --lradj 'TST' \
                --pct_start 0.2 \
                "
        )

        end_time_train = time.time()

        exec_cmd(
            f"cd src/patchtst_supervised && python3 run_longExp.py \
                --sample_size {df.shape[0]} \
                --is_training 0 \
                --data custom \
                --freq s \
                --root_path {os.path.split(current_config.raw_csv_file)[0]} \
                --data_path {os.path.split(current_config.raw_csv_file)[1]} \
                --work_folder {work_folder} \
                --checkpoints {os.path.join(work_folder, 'checkpoints')} \
                --model_id PatchTST_336_96 \
                --model PatchTST \
                --features M \
                --seq_len 336 \
                --pred_len 96 \
                --enc_in {df.shape[1]-1} \
                --des 'Exp' \
                --data_split '0.7,0.3,0.0' \
                --skip_testing True \
                --target {target_col} \
                --itr 1 \
                --batch_size 32 \
                --learning_rate 0.0001 \
                --e_layers 3 \
                --n_heads 16 \
                --d_model 128 \
                --d_ff 256 \
                --dropout 0.2 \
                --fc_dropout 0.2 \
                --head_dropout 0 \
                --patch_len 16 \
                --stride 8 \
                --patience 10 \
                --lradj 'TST' \
                --pct_start 0.2 \
                "
        )

        syn_npy = np.load(os.path.join(
            work_folder, 
            'results', 
            'PatchTST_336_96_PatchTST_custom_ftM_sl336_ll48_pl96_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0', 
            'pred.npy'))[0]

    # sanity check for NaN values
    if model_name in ['crossformer', 'd3vae', 'scinet', 'dlinear', 'patchtst']:
        if np.any(np.isnan(syn_npy)):
            raise ValueError("The synthetic data array contains NaN values.")
    
    if model_name in ['crossformer', 'd3vae']:
        print(syn_npy.shape)
        # print(syn_npy[0])
        with open(current_config.column_info_file, 'r') as f:
            column_info = json.load(f)
        print(column_info)
        print(column_info.keys())
        num_numeric_columns = len(column_info['numeric_columns'])
        num_discrete_columns_transformed = len(column_info['discrete_columns_transformed'])
        syn_df_numeric = pd.DataFrame(syn_npy[:, :num_numeric_columns], columns=column_info['numeric_columns'])

        enc = joblib.load(current_config.encoder_file)
        syn_df_discrete = enc.inverse_transform(syn_npy[:, num_numeric_columns:])
        syn_df = pd.concat([syn_df_numeric, pd.DataFrame(syn_df_discrete, columns=column_info['discrete_columns'])], axis=1)
        
        print(syn_df.shape)
        print(syn_df.head())
    
    elif model_name in ['scinet', 'dlinear', 'patchtst']: # target appended as the last column
        print(syn_npy.shape)
        # print(syn_npy[0])
        with open(current_config.column_info_file, 'r') as f:
            column_info = json.load(f)
        print(column_info)
        print(column_info.keys())
        num_numeric_columns = len(column_info['numeric_columns'])
        num_discrete_columns_transformed = len(column_info['discrete_columns_transformed'])

        if dataset_type == "pcap":
            target_col = "pkt_len"
        elif dataset_type == "netflow":
            target_col = "byt"
        column_info['numeric_columns'].remove(target_col)

        syn_df_numeric = pd.DataFrame(syn_npy[:, :num_numeric_columns-1], columns=column_info['numeric_columns'])

        enc = joblib.load(current_config.encoder_file)
        syn_df_discrete = enc.inverse_transform(syn_npy[:, num_numeric_columns-1:-1])
        syn_df = pd.concat([
                syn_df_numeric, 
                pd.DataFrame(syn_df_discrete, columns=column_info['discrete_columns']),
                pd.DataFrame(syn_npy[:, -1], columns=[target_col])
            ], 
            axis=1)
        print(syn_df.shape)
        print(syn_df.head())

    # Add first timestamp to each timeDelta if that's the case
    if model_name in ['crossformer', 'd3vae', 'scinet', 'dlinear', 'patchtst']:
        if 'timeDelta' in current_config.raw_csv_file:
            with open(current_config.column_info_file, 'r') as f:
                column_info = json.load(f)
                if dataset_type == "netflow":
                    time_col = "ts"
                elif dataset_type == "pcap":
                    time_col = "time"
                else:
                    raise ValueError(f"Unknown dataset type: {dataset_type}")
                syn_df[time_col] = syn_df[time_col] * (10**6) + column_info['first_timestamp']

    #===========================================================================
    #=================Postprocess synthetic data================================
    #===========================================================================
    # Convert bit columns to decimal columns
    def csv_bit2decimal(input_df):
        df = input_df.copy(deep=True)
        srcip_cols = df.loc[:, [f"srcip_{31-i}" for i in range(32)]]
        srcip_decimal = srcip_cols.apply(lambda x: int(''.join(x.astype(str)), 2), axis=1)
        df["srcip"] = srcip_decimal

        dstip_cols = df.loc[:, [f"dstip_{31-i}" for i in range(32)]]
        dstip_decimal = dstip_cols.apply(lambda x: int(''.join(x.astype(str)), 2), axis=1)
        df["dstip"] = dstip_decimal

        srcport_cols = df.loc[:, [f"srcport_{15-i}" for i in range(16)]]
        srcport_decimal = srcport_cols.apply(lambda x: int(''.join(x.astype(str)), 2), axis=1)
        df["srcport"] = srcport_decimal

        dstport_cols = df.loc[:, [f"dstport_{15-i}" for i in range(16)]]
        dstport_decimal = dstport_cols.apply(lambda x: int(''.join(x.astype(str)), 2), axis=1)
        df["dstport"] = dstport_decimal

        df = df.drop(columns=([f"srcip_{31-i}" for i in range(32)]+[f"dstip_{31-i}" for i in range(32)]+[f"srcport_{15-i}" for i in range(16)]+[f"dstport_{15-i}" for i in range(16)]))

        return df

    if model_name in ['ctgan', 'tvae', 'tabddpm', 'crossformer', 'd3vae', 'scinet', 'dlinear', 'patchtst']:
        syn_df = csv_bit2decimal(syn_df)
        print("Converted bits to decimals:", syn_df.shape, syn_df.columns)
        print(syn_df.head())
    
    # Sanity check and append constant columns to synthetic dataframe
    def check_and_fix_syndf(syn_df, type=None, order_by_timestamp=False):
        if syn_df['srcip'].dtype != int or syn_df['dstip'].dtype != int:
            raise ValueError("srcip and dstip should be int")
        if not pd.api.types.is_string_dtype(syn_df['proto'].dtype):
            raise ValueError(f"proto should be str")
        
        # append constant columns
        if type == 'pcap':
            syn_df['version'] = 4
            syn_df['ihl'] = 5
            for col in ['time', 'pkt_len', 'tos', 'id', 'flag', 'off', 'ttl']:
                syn_df[col] = syn_df[col].astype(int)
            syn_df = syn_df[['srcip', 'dstip', 'srcport', 'dstport', 'proto', 'time', 'pkt_len', 'version', 'ihl', 'tos', 'id', 'flag', 'off', 'ttl']] # reorder columns, important for bert inference
            time_col = 'time'
        
        elif type == 'netflow':
            # TODO: integer a few columns before exporting
            for col in ['ts', 'pkt', 'byt']:
                syn_df[col] = syn_df[col].astype(int)
            if 'label' in syn_df.columns and 'type' in syn_df.columns:
                syn_df = syn_df[['srcip', 'dstip', 'srcport', 'dstport', 'proto', 'ts', 'td', 'pkt', 'byt', 'label', 'type']]
            if 'label' not in syn_df.columns and 'type' in syn_df.columns:
                syn_df = syn_df[['srcip', 'dstip', 'srcport', 'dstport', 'proto', 'ts', 'td', 'pkt', 'byt', 'type']]
            time_col = 'ts'
        else:
            raise ValueError(f"Unknown type: {type}")
    
        if order_by_timestamp:
            syn_df = syn_df.sort_values(by=[time_col])

        return syn_df

    # Export synthetic csv to the target folder
    syn_df = check_and_fix_syndf(syn_df, type=dataset_type, order_by_timestamp=args.order_csv_by_timestamp)
    syn_df.to_csv(os.path.join(RESULT_PATH[args.config_partition]['csv'], f'{model_name}_{dataset_name}_{cur_time}.csv'), index=False)
    print("Synthetic csv exported to:", os.path.join(RESULT_PATH[args.config_partition]['csv'], f'{model_name}_{dataset_name}_{cur_time}.csv'))

    #===========================================================================
    #===========================BERT EMBEDDINGS=================================
    #===========================================================================
    # Export running time
    end_time = time.time()
    time_elapsed = end_time - start_time
    with open(os.path.join(RESULT_PATH[args.config_partition]['time'], f'{model_name}_{dataset_name}_{cur_time}.txt'), 'w') as f:
        f.write(f"{time_elapsed:.2f} seconds\n")
        f.write(f"{time_elapsed / 3600:.2f} hours\n")
        f.write(f"start_time: {datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        if end_time_train is not None:
            f.write(f"end_time_train: {datetime.datetime.fromtimestamp(end_time_train).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"end_time: {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.export_bert_embeddings:
        # Convert csv to txt for BERT inference
        # TODO: poor path handling
        if dataset_type == 'pcap':
            column_list_str = "srcip dstip srcport dstport proto time pkt_len version ihl tos id flag off ttl"
        elif dataset_type == "netflow":
            column_list_str = "srcip dstip srcport dstport proto ts td pkt byt"
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        exec_cmd(
            f"cd ../eval/fid && python3 rtf_tokenizer.py \
            --raw_csv_file {os.path.join('../', RESULT_PATH[args.config_partition]['csv'], f'{model_name}_{dataset_name}_{cur_time}.csv')} \
            --input_csv_type {dataset_type} \
            --column_list {column_list_str} \
            --export_transformed_df_only \
            --tokenizer_folder {os.path.join('../', RESULT_PATH[args.config_partition]['txt'])} \
            --transformed_df_file {model_name}_{dataset_name}_{cur_time}.txt"
        )
        print("Synthetic csv converted to txt file for BERT inference:", os.path.join(RESULT_PATH[args.config_partition]['txt'], f'{model_name}_{dataset_name}_{cur_time}.txt'))

        # Run BERT inference and export embeddings
        if dataset_type == 'pcap':
            bert_model_path = "test-mlm-bert-base-uncased-pcap-3M-mixed"
        elif dataset_type == "netflow":
            bert_model_path = "test-mlm-bert-base-uncased-netflow-3M-mixed"
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        exec_cmd(
            f"cd ../eval/fid && python3 token2embedding.py \
            --model_name_or_path {bert_model_path} \
            --input_txt_file {os.path.join('../', RESULT_PATH[args.config_partition]['txt'], f'{model_name}_{dataset_name}_{cur_time}.txt')} \
            --output_embedding_file {os.path.join('../', RESULT_PATH[args.config_partition]['npz'], f'{model_name}_{dataset_name}_{cur_time}.npz')}"
        )
        print("BERT embeddings exported to", os.path.join(RESULT_PATH[args.config_partition]['npz'], f'{model_name}_{dataset_name}_{cur_time}.npz'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--config_partition', type=str, help="Partiton of the config (e.g., small-scale, large-scale)")
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--cur_time', type=int, default=None, help='Unix timestamp representing current time') # specify a special time for debugging/testing
    parser.add_argument('--order_csv_by_timestamp', action='store_true', help='Whether to order the synthetic csv by timestamp')
    parser.add_argument('--export_bert_embeddings', action='store_true', help='Whether to export BERT embeddings')

    args = parser.parse_args()
    main(args)
