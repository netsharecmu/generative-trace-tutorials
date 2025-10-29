import argparse
import ipaddress

import pandas as pd

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input file")
    parser.add_argument("--output", type=str, required=True, help="output file")
    opt = parser.parse_args()

    df = pd.read_csv(opt.input)

    df["sa"] = df["srcip"].apply(lambda x: str(ipaddress.ip_address(x)))
    df["da"] = df["dstip"].apply(lambda x: str(ipaddress.ip_address(x)))

    df = df.rename(columns={'srcport': 'sp', 'dstport': 'dp', 'proto': 'pr'})

    df = df[["time", "pkt_len", "tos", "id", "off", "ttl", "sp", "dp", "sa", "da", "pr", "flag"]]

    df.to_csv(opt.output)