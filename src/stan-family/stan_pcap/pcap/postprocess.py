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

    start_time = int(min(df_raw["time"]))

    df_syn = pd.read_csv(opt.input)
    df_syn["time_delta"] = df_syn["time_delta"] * 1e6
    df_syn["time_delta"] = df_syn["time_delta"].cumsum()
    df_syn["time"] = df_syn["time_delta"] + start_time
    df_syn["time"] = df_syn["time"].astype(int)

    df_syn['sa'] = df_syn['sa'].apply(lambda x: int(ipaddress.IPv4Address(x)))
    df_syn['da'] = df_syn['da'].apply(lambda x: int(ipaddress.IPv4Address(x)))

    df_syn = df_syn[["time", "pkt_len", "tos", "id", "off", "ttl", "sp", "dp", "sa", "da", "pr", "flag"]]
    df_syn.rename(columns = {'sa':'srcip', 'ds':'dstip', "sp": "srcport", "dp": "dstport", "pr": "proto"}, inplace = True)

    df_syn.to_csv(opt.output)