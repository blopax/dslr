import pandas as pd
import argparse

import describe_utils
import utils


def get_clean_series(serie_name):
    return pd.Series(data[serie_name]).dropna()


def get_infos(df, show_full=False):
    infos_dict = {}
    for item in utils.SUBJECT:
        serie = df[item].dropna()
        item_describe = describe_utils.describe_serie(serie, show_full)
        infos_dict[item] = item_describe
    infos_df = pd.DataFrame(infos_dict)
    if not show_full:
        infos_df.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    else:
        infos_df.index = ['Count', 'Mean', 'Std', 'Min', '1%', '10%', '25%', '50%', '75%', '90%', '99%', 'Max']
    return infos_df


def describe(df, show_full=False):
    description = get_infos(df, show_full=show_full)
    print(description)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("-f", "--full", action="store_true",
                        help="Add more info to description.\n")
    parser.add_argument("-c", "--compare", action="store_true",
                        help="Add more info to description.\n")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data = pd.read_csv(args.dataset_file)
    if args.full:
        describe(data, show_full=True)
    else:
        describe(data)
    if args.compare:
        print(data[utils.SUBJECT].describe())
