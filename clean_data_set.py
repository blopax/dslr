import pandas as pd

import utils
import describe


def fill_nan_house_mean(df):
    for feature in utils.SELECTED_FEATURES:
        mean = (df[feature]).median()
        nan_filter = (df[feature]).isnull()
        df.loc[nan_filter, feature] = mean
    return df



def clean_df(df, selected_features=utils.SELECTED_FEATURES, train=True, train_size=0.8):
    train_size = train_size * len(df)
    if train is True:
        cleaned_df = df.loc[:train_size, ([utils.OUTPUT_COLUMN] + selected_features)]
    else:
        cleaned_df = df.loc[:, ([utils.OUTPUT_COLUMN] + selected_features)]
    out = None
    if train is True:
        cleaned_df.dropna(inplace=True)
        cleaned_df[utils.Y_COLUMN] = cleaned_df[utils.OUTPUT_COLUMN].apply(lambda x: utils.HOUSES.index(x))
        cleaned_df.reset_index(inplace=True, drop=True)
        out = cleaned_df[utils.Y_COLUMN].values.reshape(len(cleaned_df), 1)
    cleaned_df["Ones"] = 1.0
    features = cleaned_df[["Ones"] + selected_features]
    return features, out


def normalize_features(features):
    for feature_name in list(set(features.columns) - {"Ones"}):
        data = pd.read_csv("dataset_train.csv")
        infos = describe.get_infos(data)
        mean = infos.loc["Mean", feature_name]
        std = infos.loc["Std", feature_name]
        if std != 0:
            features[feature_name] -= mean
            features[feature_name] /= std
    return features
