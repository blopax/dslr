import pandas as pd
import copy

import binary_classifier
import utils
import clean_data_set


def get_thetas_train(df, selected_features=utils.SELECTED_FEATURES, alpha=0.1, epsilon=0.0001, reg_param=100,
                     mode="gradient", batch_size=1, iterations=10):
    features, output = clean_data_set.clean_df(df, selected_features=selected_features, train=True)
    thetas_init = pd.Series([0.0] * features.shape[1]).values.reshape(features.shape[1], 1)
    theta_dict, cost_list_dict = dict(), dict()
    for index, house in enumerate(utils.HOUSES):
        house_output = copy.deepcopy(output)
        house_output[house_output == index] = -1
        house_output[house_output >= 0] = 0
        house_output[house_output == -1] = 1
        if mode == "gradient":
            theta_dict[house], cost_list_dict[house] = copy.deepcopy(
                binary_classifier.gradient_descent(features, house_output, thetas_init, alpha, epsilon, reg_param))
        elif mode in ["stochastic", "mini_batch"]:
            theta_dict[house], cost_list_dict[house] = copy.deepcopy(
                binary_classifier.stochastic_descent(features, house_output, thetas_init, alpha, reg_param, batch_size,
                                                     iterations))
    return theta_dict, cost_list_dict


def train(df, selected_features=utils.SELECTED_FEATURES, alpha=0.1, epsilon=0.0001, reg_param=100, train_size=0.8,
          mode="gradient", batch_size=None, iterations=None):
    train_df = df.sample(frac=train_size, random_state=7)
    test_df = df.drop(train_df.index)

    thetas_dict, cost_list_dict = get_thetas_train(train_df, selected_features=selected_features, alpha=alpha,
                                                   epsilon=epsilon, reg_param=reg_param, mode=mode,
                                                   batch_size=batch_size, iterations=iterations)
    prediction = predict(test_df, thetas_dict)
    truth = test_df[["Index", "Hogwarts House"]]
    accuracy = utils.get_accuracy(prediction, truth, mode="simple")
    return thetas_dict, cost_list_dict, accuracy


def predict_item(x, theta_dico):
    predictions = dict()
    for index, house in enumerate(utils.HOUSES):
        predictions[house] = binary_classifier.hypothesis(x.transpose(), theta_dico[house])
    return max(predictions, key=predictions.get)


def predict(df, theta_dict):
    features, _ = clean_data_set.clean_df(df, train=False)
    prediction = features.apply(lambda x: predict_item(x, theta_dict), axis=1)
    return prediction


if __name__ == "__main__":
    train_data = pd.read_csv(utils.TRAIN_FILE)
    test_data = pd.read_csv(utils.TEST_FILE, delimiter=',')

    # with open("comb_train", mode="a+") as fd:
    #     for select_feat in utils.COMBINATORY:
    #         print(select_feat)
    #         final_theta_dict, _, train_accuracy = train(train_data, selected_features=select_feat, alpha=1,
    #                                                     epsilon=0.01, reg_param=100, train_size=0.75)
    #         print("Train accuracy is:{} for select feature {}".format(train_accuracy, select_feat), file=fd)

    final_theta_dict, _, train_accuracy = train(train_data, alpha=1, epsilon=0.01, reg_param=100, train_size=0.7)

    print("Train accuracy is: ", train_accuracy)

    test_truth = pd.read_csv("files/dataset_truth.csv")
    test_prediction = predict(test_data, final_theta_dict)

    test_accuracy = utils.get_accuracy(test_prediction, test_truth, mode='simple')
    print("Test accuracy is: ", test_accuracy)
