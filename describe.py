import pandas as pd
import numpy as np

from describe_utils import get_count_mean, get_dispertion, get_std
import utils


def get_clean_series(serie_name):
    return pd.Series(data[serie_name]).dropna()


def describe(data):
    features = []
    index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    for column in data.columns:
        if pd.Series(data[column]).dropna().dtype == np.float64:
            features.append(column)
    describe_df = pd.DataFrame(index=index, columns=features)
    print(describe_df)
    d = {}
    Count = []
    Mean = []
    Std = []
    Min = []
    onequart = []
    twoquart = []
    threequart = []
    Max = []
    for column in describe_df.columns:
        Count.append(get_count_mean(get_clean_series(data, column))[0])
        Mean.append(get_count_mean(get_clean_series(data, column))[1])
        Std.append(get_std(get_clean_series(data, column), Count[-1], Mean[-1]))
        Min.append(get_dispertion(get_clean_series(data, column), Count[-1])[0])
        onequart.append(get_dispertion(get_clean_series(data, column), Count[-1])[1])
        twoquart.append(get_dispertion(get_clean_series(data, column), Count[-1])[2])
        threequart.append(get_dispertion(get_clean_series(data, column), Count[-1])[3])
        Max.append(get_dispertion(get_clean_series(data, column), Count[-1])[4])
    d['Count'] = Count
    d['Mean'] = Mean
    d['Std'] = Std
    d['Min'] = Min
    d['onequart'] = onequart
    d['twoquart'] = twoquart
    d['threequart'] = threequart
    d['Max'] = Max
    print(d)


if __name__ == "__main__":
    data = pd.read_csv("dataset_train.csv")
    count_black_arts = get_count_mean(get_clean_series(data, "Defense Against the Dark Arts"))[0]
    describe(data)
    print(count_black_arts)
