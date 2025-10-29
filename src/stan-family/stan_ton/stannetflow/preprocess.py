import configparser
from stannetflow.analyze_functions import analyze, extract, prepare_folders, recover_userlist_from_folder
from stannetflow import STANSynthesizer, STANCustomDataLoader, NetflowFormatTransformer, STANTemporalTransformer
import glob
import pandas as pd


def user_analysis():
  analyze()

def user_selection():
  config = configparser.ConfigParser()
  config.read('./ugr16_config.ini')
  # print({section: dict(config[section]) for section in config.sections()})
  user_list = config['DEFAULT']['userlist'].split(',')
  print('extracting:', user_list)
  prepare_folders()
  # recover_userlist_from_folder()
  extract(user_list)

def download_ugr16():
  print('Visit the following url to download april_week3.csv')  
  print('https://nesg.ugr.es/nesg-ugr16/april_week3.php')

def _prepare(folder='', output='', agg=1, dataset_stat=None):
  if len(folder) and len(output):
    count = 0
    ntt = NetflowFormatTransformer(dataset_stat)
    tft = STANTemporalTransformer(folder)
    for f in glob.glob(output):
      print('user:', f)
      this_ip = f.split("_")[-1][:-4]
      df = pd.read_csv(f)
      df['this_ip'] = this_ip
      tft.push_back(df, agg=agg, transformer=ntt)
      count += 1
    print(count)

def prepare_standata(agg=5, train_folder='stan_data/day1_data', train_output='to_train.csv', 
                      test_folder='stan_data/day2_data', test_output='to_test.csv', dataset_stat=None):
  if len(train_folder):
    print('making train for:')
    _prepare('stan_data/'+train_output, train_folder+'/*.csv', agg=agg, dataset_stat=dataset_stat)
  if len(test_folder):
    print('making test for:')
    _prepare('stan_data/'+test_output, test_folder+'/*.csv', agg=agg, dataset_stat=dataset_stat)

if __name__ == "__main__":
  # download_ugr16()
  # user_analysis()
  # user_selection()
  prepare_standata(agg=5, train_folder='../ugr_test', train_output='../to_train.csv', 
                      test_folder='../ugr_test', test_output='../to_test.csv')