import utils
import describe_utils


def clean_df(df):
    cleaned_df = df[([utils.OUTPUT_COLUMN] + utils.SELECTED_FEATURES)]
    cleaned_df.dropna(inplace=True)
    cleaned_df.reset_index(inplace=True, drop=True)
    cleaned_df["Ones"] = 1.0
    cleaned_df[utils.Y_COLUMN] = cleaned_df[utils.OUTPUT_COLUMN].astype('category').cat.codes
    return cleaned_df[["Ones"] + utils.SELECTED_FEATURES], cleaned_df[utils.Y_COLUMN].values.reshape(len(cleaned_df), 1)


def normalize_features(features):
    for feature_name in features.columns:
        count, mean = describe_utils.get_count_mean(features[feature_name])
        std = describe_utils.get_std(features[feature_name], count, mean)
        if std != 0:
            features[feature_name] -= mean
            features[feature_name] /= std
    return features
