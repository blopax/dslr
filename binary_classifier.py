import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils
import clean_data_set


def hypothesis(x, thetas):
    theta_x = np.dot(x, thetas)
    return 1.0 / (1.0 + np.exp(-theta_x))


def cost_function(x, y, thetas):
    h = hypothesis(x, thetas)
    mini, maxi = h.min(), h.max()
    return 1 / len(y) * (-np.dot(y.transpose(), np.log(h)) - np.dot((1-y).transpose(), np.log(1 - h)))


def update_thetas(x, y, thetas, alpha):
    m = len(y)
    h = hypothesis(x, thetas)
    thetas -= alpha / m * np.dot(x.transpose(), h - y)
    return thetas


def gradient_descent(x, y, thetas, alpha=0.1, epsilon=0.0001):
    cost_list = [cost_function(x, y, thetas)]
    while len(cost_list) == 1 or cost_list[-1] / cost_list[-2] <= 1 - epsilon:
        thetas = update_thetas(x, y, thetas, alpha)
        cost_list.append(float(cost_function(x, y, thetas)))
    return thetas, cost_list


if __name__ == "__main__":
    df = pd.read_csv(utils.TRAIN_FILE)
    features, output = clean_data_set.clean_df(df)
    features = clean_data_set.normalize_features(features)
    thetas_init = pd.Series([0.0] * features.shape[1]).values.reshape(features.shape[1], 1)
    output[output != 1] = 0
    final_thetas, cost_list = gradient_descent(features, output, thetas_init)
    plt.plot([i for i in range(len(cost_list))], list(cost_list))
    plt.show()
