import pandas as pd
import copy

import binary_classifier
import utils
import clean_data_set


def train(df, selected_features=utils.SELECTED_FEATURES, alpha=0.1, epsilon=0.0001, reg_param=100, train_size=0.8):
    features, output = clean_data_set.clean_df(df, selected_features=selected_features, train=True, train_size=train_size)
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


def predict_item(x, theta_dico):
    predictions = dict()
    for index, house in enumerate(utils.HOUSES):
        predictions[house] = binary_classifier.hypothesis(x.transpose(), theta_dico[house])
    return max(predictions, key=predictions.get)


def predict(df, theta_dict):
    features, _ = clean_data_set.clean_df(df, train=False)
    prediction = features.apply(lambda x: predict_item(x, theta_dict), axis=1)
    return prediction


def accuracy(prediction, truth, mode='simple'):
    errors_loc = truth["Hogwarts House"] != prediction
    errors = pd.DataFrame()
    errors["truth"] = truth[errors_loc]["Hogwarts House"]
    errors["prediction"] = prediction[errors_loc]

    total_accuracy = 1.0 - len(errors) / len(truth)

    house_acc = dict()
    for house in utils.HOUSES:
        house_acc[house] = 1 - len(errors[errors["truth"] == house]) / len(truth[truth["Hogwarts House"] == house])
    if mode == 'all':
        print("Zoom on wrong predictions:\n", errors)
        print("Accuracy per house: ", house_acc)
    print("Total accuracy is: ", total_accuracy)


if __name__ == "__main__":
    train_df = pd.read_csv(utils.TRAIN_FILE)
    test_df = pd.read_csv("dataset_test.csv", delimiter=',')

    # test_df = pd.read_csv(utils.TRAIN_FILE)
    # test_df = test_df.loc[1280:, :]
    # print(test_df.shape)

    # for select_feat in utils.COMBINATORY:
    #     final_theta_dict = train(train_df, selected_features=select_feat, alpha=1, epsilon=0.01, reg_param=100,
    #                              fill_mean=False)
    #     feat, out = clean_data_set.clean_df(test_df, selected_features=select_feat, train=False)
    #     feat = clean_data_set.normalize_features(feat)
    #     prediction = feat.apply(lambda x: predict(x, final_theta_dict), axis=1)
    #     truth = pd.read_csv("dataset_truth.csv")
    #     print(select_feat)
    #     accuracy(prediction, truth, mode='all')

    final_theta_dict = train(train_df, alpha=1, epsilon=0.01, reg_param=100, train_size=0.8)
    # feat = clean_data_set.normalize_features(feat)
    truth = pd.read_csv("dataset_truth.csv")
    # truth = test_df.loc[:, ["Index", "Hogwarts House"]]
    prediction = predict(test_df, final_theta_dict)

    accuracy(prediction, truth, mode='all')
    # print(final_theta_dict)
