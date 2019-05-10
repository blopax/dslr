import pandas as pd
import copy

import binary_classifier
import utils
import clean_data_set


def train(df, selected_features=utils.SELECTED_FEATURES, alpha=0.1, epsilon=0.0001, reg_param=100, fill_mean=False):
    if fill_mean:
        df = clean_data_set.fill_nan_house_mean(df)

    features, output = clean_data_set.clean_df(df, selected_features=selected_features, train=True)
    features = clean_data_set.normalize_features(features)
    thetas_init = pd.Series([0.0] * features.shape[1]).values.reshape(features.shape[1], 1)
    theta_dict = dict()
    for index, house in enumerate(utils.HOUSES):
        house_output = copy.deepcopy(output)
        house_output[house_output == index] = -1
        house_output[house_output >= 0] = 0
        house_output[house_output == -1] = 1
        theta_dict[house], cost_list = copy.deepcopy(
            binary_classifier.gradient_descent(features, house_output, thetas_init, alpha, epsilon, reg_param))

    return theta_dict


def predict(x, theta_dico):
    predictions = dict()
    for index, house in enumerate(utils.HOUSES):
        predictions[house] = binary_classifier.hypothesis(x.transpose(), theta_dico[house])
    return max(predictions, key=predictions.get)


if __name__ == "__main__":
    train_df = pd.read_csv(utils.TRAIN_FILE)
    # print(train_df[utils.SELECTED_SUBJECT].describe())
    test_df = pd.read_csv("dataset_test.csv", delimiter=',')

    #
    # for each in utils.COMBINATORY:
    #     final_theta_dict = train(train_df, selected_feat=each, alpha=1, epsilon=0.01, reg_param=100, fill_mean=False)
    #     feat, out = clean_data_set.clean_df(test_df, selected_feat=each, train=False)
    #     feat = clean_data_set.normalize_features(feat)
    #     feat = feat
    #     prediction = feat.apply(lambda x: predict(feat, final_theta_dict), afeatis=1)
    #     truth = pd.read_csv("dataset_truth.csv")
    #     house_acc = dict()
    #     pb_loc = truth["Hogwarts House"] != prediction
    #     show = pd.DataFrame()
    #     show["truth"] = truth[pb_loc]["Hogwarts House"]
    #     show["prediction"] = prediction[pb_loc]
    #     print(each)

    final_theta_dict = train(train_df, alpha=1, epsilon=0.01, reg_param=100, fill_mean=False)
    feat, out = clean_data_set.clean_df(test_df, train=False)
    feat = clean_data_set.normalize_features(feat)
    feat = feat
    prediction = feat.apply(lambda x: predict(feat, final_theta_dict), afeatis=1)
    truth = pd.read_csv("dataset_truth.csv")
    house_acc = dict()
    pb_loc = truth["Hogwarts House"] != prediction
    show = pd.DataFrame()
    show["truth"] = truth[pb_loc]["Hogwarts House"]
    show["prediction"] = prediction[pb_loc]
    print(show)
    # print(truth["Hogwarts House"][pb_loc].value_counts())
    # print(prediction[pb_loc].value_counts())
    # for house in utils.HOUSES:
    # truth_house = truth[(truth["Hogwarts House"] == prediction) (truth["Hogwarts House"] == house)]
    # house_acc[house] = len(truth_house[truth_house == True] / len(truth_house))
    # print(house_acc)
    print(final_theta_dict)
    print(len(truth[truth["Hogwarts House"] == prediction]) / len(truth))
