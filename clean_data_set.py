import pandas as pd

import utils
import describe


def fillna_house_mean(df):
    filtered_df = df
    for feature in utils.SELECTED_FEATURES:
        mean = (df[feature]).median()
        filter = (df[feature]).isnull()
        df.loc[filter, feature] = mean
        # (df[feature]).fillna(mean)

        # mean_house = dict()
        # for house in utils.HOUSES:
        #     serie = df[feature][df["Hogwarts House"] == house]
        #     mean = serie.median()
        #     # print(feature, house, mean)
        #     house = df["Hogwarts House"] == house
        #     na = (df[feature]).isnull()
        #     filter = house & na
        #     # print(filter.value_counts(), mean)
        #     # print(df.loc[filter, feature])
        #     df.loc[filter, feature] = mean

    # print("bob", df["Divination"].head(60))
    # print(df[utils.SELECTED_FEATURES].head())
    # print(df[utils.SELECTED_FEATURES].describe())
    return df


def clean_df(df, sf, train=True):
    cleaned_df = df[([utils.OUTPUT_COLUMN] + sf)]
    out = None
    if train is True:
        cleaned_df.dropna(inplace=True)
        cleaned_df[utils.Y_COLUMN] = cleaned_df[utils.OUTPUT_COLUMN].apply(lambda x: utils.HOUSES.index(x)) #astype('category').cat.codes
        cleaned_df.reset_index(inplace=True, drop=True)
        out = cleaned_df[utils.Y_COLUMN].values.reshape(len(cleaned_df), 1)
    cleaned_df["Ones"] = 1.0
    features = cleaned_df[["Ones"] + sf]
    return features, out


def normalize_features(features):
    for feature_name in list(set(features.columns) - set(["Ones"])):
        data = pd.read_csv("dataset_train.csv")
        infos = describe.get_infos(data)
        mean = infos.loc["Mean", feature_name]
        std = infos.loc["Std", feature_name]
        if std != 0:
            features[feature_name] -= mean
            features[feature_name] /= std
    return features
