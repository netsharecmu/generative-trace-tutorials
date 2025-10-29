# This only works on small datasets

import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input)

df = df[["Date first seen", "Duration", "Src IP Addr", "Dst IP Addr", "Src Pt", "Dst Pt",
        "Proto", "Flags", "Packets", "Bytes", "class", "attackType"]]

df = df.rename(columns={"Date first seen": "ts", "Duration": "td", "Src IP Addr": "sa",
                    "Dst IP Addr": "da", "Src Pt": "sp", "Dst Pt": "dp", "Proto": "pr", "Flags": "flg",
                    "Packets": "pkt", "Bytes": "byt"})

df.to_csv(args.output, index=False)
