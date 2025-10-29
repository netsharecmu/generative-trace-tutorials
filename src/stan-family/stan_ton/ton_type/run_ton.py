import argparse
import json
import os
from os.path import exists

import numpy as np
import pandas as pd
from stannetflow import (NetflowFormatTransformer, STANCustomDataLoader,
                         STANSynthesizer)
from stannetflow.preprocess import prepare_standata
from tqdm import tqdm


def preprocess(host_ips_prefix = None, raw_file_path = None, out_file_dir = None):
    dataset_stat = {}

    pd.options.mode.chained_assignment = None

    df = pd.read_csv(raw_file_path)
    print(df)

    df = df.sort_values(by=["te"])
    # df["teT"] = (pd.to_datetime(df["ts"]).astype(int) / 10**9 / 3600).astype(int)
    df["teT"] = pd.to_datetime(df["te"]).dt.hour
    df["teDelta"] = pd.to_datetime(df["te"]).diff().dt.total_seconds()
    df["teDelta"].iloc[0] = 0


    df = df[["teT", "teDelta", "byt", "pkt", "td", "sp", "dp", "sa", "da", "pr", "label", "type"]]

    print(df)
    print(len(df))

    host_ips = {}

    for row_idx in tqdm(range(len(df))):
      srcip = df.at[row_idx, "sa"]
      dstip = df.at[row_idx, "da"]

      key = None
      for prefix in host_ips_prefix:
        if srcip.startswith(prefix):
          key = srcip
          break
        if dstip.startswith(prefix):
          key = dstip
          break
      
      if key == None:
        print("Cannot find a host ip candidate from this row of data! Continue on the next one")
        continue

      if key not in host_ips:
        host_ips[key] = [list(df.iloc[row_idx])]
      else:
        host_ips[key].append(list(df.iloc[row_idx]))

    for key in host_ips.keys():
      host_ips[key] = pd.DataFrame(host_ips[key], columns=["teT", "teDelta", "byt", "pkt", "td", "sp", "dp", "sa", "da", "pr", "label", "type"])
      out_file_path = f"{out_file_dir}/modofied_{key}.csv"
      host_ips[key].to_csv(out_file_path, index=False)

    teT_max = df["teT"].max()
    teDelta_max = df["teDelta"].max()
    td_max = df["td"].max()
    pkt_max = np.log(df["pkt"]).max()
    byt_max = np.log(df["byt"]).max()

    dataset_stat = {
        "teT_max": teT_max,
        "teDelta_max": teDelta_max,
        "td_max": td_max,
        "pkt_max": pkt_max,
        "byt_max": byt_max,
        "host_ips": list(host_ips.keys())
    }

    print(f"\tteT_max: \033[94m{teT_max}\033[0m\n\
        \tteDelta_max: \033[94m{teDelta_max}\033[0m\n\
        \ttd_max: \033[94m{td_max}\033[0m\n\
        \tpkt_max(log): \033[94m{pkt_max}\033[0m\n\
        \tbyt_max(log): \033[94m{byt_max}\033[0m")

    return dataset_stat


def run(train_file, dataset_stat, output=None, sample_num=1000000, load_checkpoint=False):
  train_loader = STANCustomDataLoader(train_file, 6, 28).get_loader()
  ugr16_n_col, ugr16_n_agg, ugr16_arch_mode = 28, 5, 'B'
  # index of the columns that are discrete (in one-hot groups), categorical (number of types)
  # or any order if wanted
  ugr16_discrete_columns = [[11,12], [13, 14, 15], [16, 17], [18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
  ugr16_categorical_columns = {5:1670, 6:1670, 7:256, 8:256, 9:256, 10:256}
  ugr16_execute_order = [0,1,13,11,18,16,5,6,7,8,9,10,3,2,4]

  stan = STANSynthesizer(dim_in=ugr16_n_col, dim_window=ugr16_n_agg, 
          discrete_columns=ugr16_discrete_columns,
          categorical_columns=ugr16_categorical_columns,
          execute_order=ugr16_execute_order,
          arch_mode=ugr16_arch_mode
          )
  
  if load_checkpoint is False:
    stan.batch_fit(train_loader, epochs=100)
  else:
    stan.load_model('ep99') # checkpoint name
    # validation
    # stan.validate_loss(test_loader, loaded_ep='ep998')

    ntt = NetflowFormatTransformer(dataset_stat=dataset_stat)
    samples = stan.sample(sample_num, dataset_stat)
    df_rev = ntt.rev_transfer(samples)
    df_rev.to_csv(output)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__=="__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=str, required=True, help="input file")
  parser.add_argument("--generate", type=bool, default=False, help="generate")
  parser.add_argument("--output", type=str, default="False", help="input file")
  parser.add_argument("--sample_num", type=int, default=100000, help="input file")
  opt = parser.parse_args()

  print("please be sure that the host ip prefix has been provided correctly")
#   host_ips_prefix = ["42.219."]
  host_ips_prefix = ["192.168."]
  os.makedirs('stan_data', exist_ok=True)
  os.makedirs('stan_data/netflow', exist_ok=True)


  stat_path = "./stat.txt"
  file_exists = exists(stat_path)
  if file_exists == True:
    with open(stat_path) as f:
      dataset_stat = json.load(f)
  else: 
    dataset_stat = preprocess(host_ips_prefix=host_ips_prefix,
                        raw_file_path=opt.input,
                        out_file_dir='stan_data/netflow')
    prepare_standata(agg=5, train_folder='stan_data/netflow', train_output='to_train.csv', 
                  test_folder='stan_data/netflow', test_output='to_test.csv', dataset_stat=dataset_stat)
    with open(stat_path, 'w') as fp:
        json.dump(dataset_stat, fp, cls=NpEncoder)
  
  run("stan_data/to_train.csv", dataset_stat, opt.output, opt.sample_num, load_checkpoint=opt.generate)
