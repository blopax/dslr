import pandas as pd
import copy
import matplotlib.pyplot as plt

import binary_classifier
import utils
import clean_data_set


def train(df, alpha=0.1, epsilon=0.0001, fill_mean=False):
    if fill_mean:
        df = clean_data_set.fillna_house_mean(df)
    print("in Train")
    print(df[utils.SELECTED_FEATURES].describe())
    print(df[utils.SELECTED_FEATURES].head())

    features, output = clean_data_set.clean_df(df)
    features = clean_data_set.normalize_features(features)
    print(features.shape)
    thetas_init = pd.Series([0.0] * features.shape[1]).values.reshape(features.shape[1], 1)
    theta_dict = dict()
    for index, house in enumerate(utils.HOUSES):
        house_output = copy.deepcopy(output)
        house_output[house_output == index] = -1
        house_output[house_output >= 0] = 0
        house_output[house_output == -1] = 1
        theta_dict[house], cost_list = copy.deepcopy(binary_classifier.gradient_descent(features, house_output, thetas_init, alpha, epsilon))
        # plt.plot([i for i in range(len(cost_list))], list(cost_list))
        # plt.show()

    return theta_dict


def predict(x, theta_dico):
    predictions = dict()
    for index, house in enumerate(utils.HOUSES):
        predictions[house] = binary_classifier.hypothesis(x.transpose(), theta_dico[house])
    return max(predictions, key=predictions.get)


if __name__ == "__main__":
    train_df = pd.read_csv(utils.TRAIN_FILE)
    print(train_df[utils.SELECTED_SUBJECT].describe())
    test_df = pd.read_csv("dataset_test.csv", delimiter=',')
    theta_dico = train(train_df, alpha=1, epsilon=0.01, fill_mean=False)
    features, out = clean_data_set.clean_df(test_df, train=False)
    features = clean_data_set.normalize_features(features)
    x = features
    prediction = features.apply(lambda x: predict(x, theta_dico), axis=1)
    truth = pd.read_csv("dataset_truth.csv")
    house_acc = dict()
    pb_loc = truth["Hogwarts House"] != prediction
    show = pd.DataFrame()
    show["truth"] = truth[pb_loc]["Hogwarts House"]
    show["prediction"] = prediction[pb_loc]
    print(show)
    print(truth["Hogwarts House"][pb_loc].value_counts())
    print(prediction[pb_loc].value_counts())
    # for house in utils.HOUSES:
        # truth_house = truth[(truth["Hogwarts House"] == prediction) (truth["Hogwarts House"] == house)]
        # house_acc[house] = len(truth_house[truth_house == True] / len(truth_house))
    # print(house_acc)
    print(len(truth[truth["Hogwarts House"] == prediction]) / len(truth))

