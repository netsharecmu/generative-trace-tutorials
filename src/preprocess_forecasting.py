import argparse
import os
import json
import joblib
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def main(args):
    print(f'Input file: {args.input_file}')
    print(f'Output file: {os.path.join(args.output_folder, args.output_csv_file)}')
    output_col_info = {}

    df = pd.read_csv(args.input_file)

    common_discrete_columns = \
        [f"srcip_{31-i}" for i in range(32)] + \
        [f"dstip_{31-i}" for i in range(32)] + \
        [f"srcport_{15-i}" for i in range(16)] + \
        [f"dstport_{15-i}" for i in range(16)] + \
        ['proto']
    time_col = None
    if args.dataset_name in ['caida', 'dc', 'ca', 'm57']:
        discrete_columns = common_discrete_columns + ['flag']
        time_col = 'time'
        dropped_columns = []
        for col in ['version', 'ihl', 'chksum']:
            if col in df.columns:
                dropped_columns.append(col)
        df.drop(columns=dropped_columns, inplace=True)
    elif args.dataset_name in ['ugr16', 'cidds', 'ton']:
        time_col = 'ts'
        discrete_columns = common_discrete_columns + ['type']
        if args.dataset_name in ['cidds', 'ton']:
            discrete_columns += ['label']
    else:
        raise ValueError(f'Unknown dataset name: {args.dataset_name}')
    numeric_columns = list(set(df.columns) - set(discrete_columns))
    
    # Convert the `time_col` column to unix timestamp and then to datetime
    df['date'] = pd.to_datetime(df[time_col] / 1000000, unit='s')

    # Format the datetime objects to string
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    enc.set_output(transform='pandas')
    enc.fit(df[discrete_columns])
    joblib.dump(enc, os.path.join(args.output_folder, args.output_encoder_file))
    ohe_encoded_df = enc.transform(df[discrete_columns])
    df.drop(discrete_columns, axis=1, inplace=True)
    df = pd.concat([df, ohe_encoded_df], axis=1)

    discrete_columns_transformed = list(enc.get_feature_names_out(input_features=discrete_columns))

    output_col_info['numeric_columns'] = numeric_columns
    output_col_info['discrete_columns'] = discrete_columns
    output_col_info['discrete_columns_transformed'] = discrete_columns_transformed

    # Make 'date' the first column
    df = df[['date'] + numeric_columns + discrete_columns_transformed]
    
    print("After one-hot encoding:", df.shape)
    print("# of numeric columns:", len(numeric_columns))
    print("# of discrete columns:", len(discrete_columns_transformed))
    # print(list(df.columns))
    assert (len(numeric_columns) + len(discrete_columns_transformed) + 1) == df.shape[1]

    df.to_csv(os.path.join(args.output_folder, args.output_csv_file), index=False)

    with open(os.path.join(args.output_folder, args.output_column_info_file), 'w') as f:
        json.dump(output_col_info, f, indent=4)


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process some files")

    # Add the parameters positional/optional
    parser.add_argument("--input_file", type=str, help="Specify the input file", required=True)
    parser.add_argument("--output_folder", type=str, help="Specify the output folder", required=True)
    parser.add_argument("--output_csv_file", type=str, help="Specify the output csv file", default="raw_bits_transformed_forecasting.csv")
    parser.add_argument("--output_column_info_file", type=str, help="Specify the output column info", default="column_info_transformed_forecasting.json")
    parser.add_argument("--output_encoder_file", type=str, help="Specify the output encoder file", default="encoder_transformed_forecasting.pkl")
    parser.add_argument("--dataset_name", type=str, help="Specify the dataset name", required=True)

    # Parse the arguments
    args = parser.parse_args()

    main(args)

