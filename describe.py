import pandas as pd

import describe_utils
import utils


def get_clean_series(serie_name):
    return pd.Series(data[serie_name]).dropna()


def get_infos(df):
    infos_dict = {}
    for item in utils.SUBJECT:
        serie = df[item].dropna()
        item_describe = describe_utils.describe_serie(serie)
        infos_dict[item] = item_describe
    infos_df = pd.DataFrame(infos_dict)
    infos_df.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    return infos_df


def describe(df):
    description = get_infos(df)
    print(description)


if __name__ == "__main__":
    data = pd.read_csv("dataset_train.csv")
    describe(data)
    print(data[utils.SUBJECT].describe())
