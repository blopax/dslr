import argparse
import numpy as np
import pandas as pd

import multiclass_classifier


def csv_to_theta_dict(file_name):
    dataframe = pd.read_csv(file_name)
    theta_dict = pd.DataFrame.to_dict(dataframe, orient='list')
    for key, value in theta_dict.items():
        theta_dict[key] = np.array(value).reshape(4, 1)
    return theta_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_test_file", help="Please add a csv file as a parameter (.csv file)", type=str)
    parser.add_argument("weights", help="Please add a csv file as a parameter (.csv file)", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # noinspection PyUnresolvedReferences
    try:
        test_data = pd.read_csv(args.dataset_test_file, delimiter=',')
        thetas_dict = csv_to_theta_dict(args.weights)
        test_prediction = multiclass_classifier.predict(test_data, thetas_dict)
        test_prediction.to_csv("houses.csv", index_label="Index", header=["Hogwarts House"])
    except FileNotFoundError as err:
        print("Error: {}".format(err))
    except pd.errors.ParserError as err:
        print("Error: dataset_train not csv or well formatted.\n{}".format(err))
