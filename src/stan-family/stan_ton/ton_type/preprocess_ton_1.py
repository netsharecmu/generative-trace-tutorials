
import pickle, ipaddress, copy, time, more_itertools, os, math, json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm
from gensim.models import Word2Vec, word2vec
from sklearn import preprocessing

def validate_ip_address(address):
    try:
        ip = ipaddress.ip_address(address)
        # print("IP address {} is valid. The object returned is {}".format(address, ip))
        return True
    except ValueError:
        # print("IP address {} is not valid".format(address)) 
        return False

def validate_ippairs(srcips, dstips):
    res = []
    for idx in range(len(srcips)):
        if validate_ip_address(srcips[idx]) and validate_ip_address(dstips[idx]):
            res.append(True)
        else:
            res.append(False)

    return res

def preprocess_netflow(raw_df):
    raw_df = raw_df[["ts","src_ip","src_port","dst_ip","dst_port","proto","duration","byt","pkt","label","type"]]

    print("raw df shape:", raw_df.shape)
    print("Start IP filtering...")
    filtered_df = raw_df[validate_ippairs(raw_df['src_ip'], raw_df['dst_ip'])].reset_index(drop=True)
    print("Finish IP filtering...")
    print("filtered df shape:", filtered_df.shape)

    print("Start transforming Bytes Field...")
    for row_idx in tqdm(range(0, len(filtered_df))):
        try:
            filtered_df.at[row_idx, 'byt'] = float(filtered_df.at[row_idx, 'byt'])
        except:
            filtered_df.at[row_idx, 'byt'] = float(filtered_df.at[row_idx, 'byt'].split()[0]) * (10**6)
        
    print("Finish transforming Bytes Field...")

    return filtered_df

def sanity_check_filtered_df(filtered_df):
    ips = list(filtered_df["Src IP Addr"]) + list(filtered_df["Dst IP Addr"])

    for ip in ips:
        if len(ip.split(".")) != 4:
            return False
    
    return True

def main(args):
    filtered_csv_filename = args.output

    if os.path.exists(filtered_csv_filename):
        print("Load existing filtered df...")
        filtered_df = pd.read_csv(filtered_csv_filename)
    
    else:
        print("Start filtering from scratch...")
        print("Reading raw csv from {}".format(args.input))
        raw_df = pd.read_csv(args.input)
        filtered_df = preprocess_netflow(raw_df)
        print("Writing filtered csv to {}".format(filtered_csv_filename))
        filtered_df.to_csv(filtered_csv_filename, index=False)

    if args.sanity_check_filtered:
        print("sanity checking filtered df...")
        if sanity_check_filtered_df(filtered_df) == True:
            print("sanity check filtered df on IPs: PASS!")
        else:
            raise ValueError("sanity check filtered df on IPs: ERROR!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--sanity_check_filtered', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
