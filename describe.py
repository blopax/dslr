import pandas as pd

import describe_utils
import utils


def get_clean_series(serie_name):
    return pd.Series(data[serie_name]).dropna()


def describe(df):
    df_dict = {}
    for item in utils.SUBJECT:
        serie = df[item].dropna()
        item_describe = describe_utils.describe_serie(serie)
        df_dict[item] = item_describe
    df = pd.DataFrame(df_dict)
    df.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    print(df)


if __name__ == "__main__":
    data = pd.read_csv("dataset_train.csv")
    describe(data)
    print(data[utils.SUBJECT].describe())
