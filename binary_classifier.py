import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

import utils
import clean_data_set


def hypothesis(x, thetas):
    theta_x = np.dot(x, thetas)
    return 1.0 / (1.0 + np.exp(-theta_x))


def cost_function(x, y, thetas, reg_param=1):
    h = hypothesis(x, thetas)
    regularization = reg_param / (2 * len(y)) * np.dot((thetas[1:]).transpose(), thetas[1:])
    return 1 / len(y) * (-np.dot(y.transpose(), np.log(h)) - np.dot((1-y).transpose(), np.log(1 - h))) + regularization


def update_thetas(x, y, thetas, alpha, reg_param=1, batch_size=None):
    if not batch_size:
        batch_size = len(y)
    h = hypothesis(x, thetas)
    reg_vector = copy.deepcopy(thetas)
    reg_vector[0] = 0

    thetas -= alpha / batch_size * (np.dot(x.transpose(), h - y) + reg_param * reg_vector)
    return thetas


def gradient_descent(x, y, thetas, alpha=0.1, epsilon=0.0001, reg_param=1):
    cost_list = [cost_function(x, y, thetas)]
    while len(cost_list) == 1 or cost_list[-1] / cost_list[-2] <= 1 - epsilon:
        thetas = update_thetas(x, y, thetas, alpha, reg_param)
        cost_list.append(float(cost_function(x, y, thetas)))
    return thetas, cost_list


def stochastic_descent(x, y, thetas, alpha=0.005, reg_param=1, batch_size=1, iterations=10):
    m = len(y)
    cost_list = [cost_function(x, y, thetas)]
    for iteration in range(iterations):
        for row in range(m // batch_size):
            start_row = row * batch_size
            finish_row = min(start_row + batch_size, m)
            x_rows = x.iloc[start_row:finish_row]
            thetas = update_thetas(x_rows, y[start_row:finish_row], thetas, alpha, reg_param, batch_size=batch_size)
            cost_list.append(float(cost_function(x, y, thetas)))
    return thetas, cost_list


if __name__ == "__main__":
    df = pd.read_csv(utils.TRAIN_FILE)
    features, output = clean_data_set.clean_df(df)
    features = clean_data_set.normalize_features(features)
    thetas_init = pd.Series([0.0] * features.shape[1]).values.reshape(features.shape[1], 1)
    output[output != 1] = 0
    final_thetas, final_cost_list = gradient_descent(features, output, thetas_init)
    plt.plot([i for i in range(len(final_cost_list))], list(final_cost_list))
    plt.show()
