import argparse
import datetime
import ipaddress
import time

import pandas as pd

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True, help="raw file")
    parser.add_argument("--input", type=str, required=True, help="input file")
    parser.add_argument("--output", type=str, required=True, help="output file")
    opt = parser.parse_args()

    df_raw = pd.read_csv(opt.raw)


    df_raw["te"] = df_raw["ts"] + df_raw["td"]
    start_time = int(min(df_raw["te"]))

    df_syn = pd.read_csv(opt.input)
    df_syn["time_delta"] = df_syn["time_delta"] * 1e6
    df_syn["time_delta"] = df_syn["time_delta"].cumsum()
    df_syn["ts"] = df_syn["time_delta"] + start_time
    df_syn["ts"] = df_syn["ts"].astype(int)

    df_syn['sa'] = df_syn['sa'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    df_syn['da'] = df_syn['da'].apply(lambda x: int(ipaddress.IPv4Address(x)))

    df_syn = df_syn[["sa","da","sp","dp","pr","ts","time_duration","pkt","byt","type"]]
    df_syn.rename(columns = {'sa':'srcip', 'ds':'dstip', "sp": "srcport", "dp": "dstport", "pr": "proto", "time_duration": "td"}, inplace = True)

    df_syn.to_csv(opt.output)