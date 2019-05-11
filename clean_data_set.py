import pandas as pd

import utils
import describe
import warnings


def normalize_features(features):
    warnings.filterwarnings("ignore")
    for feature_name in list(set(features.columns) - {"Ones"}):
        data = pd.read_csv("dataset_train.csv")
        infos = describe.get_infos(data)
        mean = infos.loc["Mean", feature_name]
        std = infos.loc["Std", feature_name]
        if std != 0:
            features[feature_name] -= mean
            features[feature_name] /= std
    return features


def get_features(df, selected_features=utils.SELECTED_FEATURES, train=True):
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


def clean_df(df, selected_features=utils.SELECTED_FEATURES, train=True):
    if not train:
        test_df = df[utils.SUBJECT]
        test = normalize_features(test_df)
        test = test.apply(lambda x: x.fillna(x.mean()), axis=1)
        features, _ = get_features(test, train=False)
        return features, None
    else:
        features, output = get_features(df, selected_features=selected_features, train=train)
        features = normalize_features(features)
        return features, output
