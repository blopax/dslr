import pandas as pd
import copy

import binary_classifier
import utils
import clean_data_set


def train(df):
    features, output = clean_data_set.clean_df(df)
    print(type(output))
    print(output, output.shape, output.sum())
    features = clean_data_set.normalize_features(features)
    thetas_init = pd.Series([0.0] * features.shape[1]).values.reshape(features.shape[1], 1)
    theta_dict = dict()
    for index, house in enumerate(utils.HOUSES):
        house_output = copy.deepcopy(output)
        house_output[house_output == index] = -1
        house_output[house_output >= 0] = 0
        house_output[house_output == -1] = 1
        print(index, house, house_output, house_output.shape, house_output.sum())
        thetas = binary_classifier.gradient_descent(features, house_output, thetas_init)[0]
        theta_dict[house] = thetas
        print(thetas)
    return theta_dict


if __name__ == "__main__":
    df = pd.read_csv(utils.TRAIN_FILE)
    theta_dict = train(df)
    print(theta_dict)
