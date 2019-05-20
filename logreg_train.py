import argparse
import pandas as pd
import matplotlib.pyplot as plt

import multiclass_classifier
import utils


def theta_dict_to_csv(theta_dict):
    formatted_theta_dict = dict()
    for key, value in theta_dict.items():
        formatted_theta_dict[key] = value.reshape(4, ).tolist()
    dataframe = pd.DataFrame.from_dict(formatted_theta_dict)
    print(dataframe)
    dataframe.to_csv("weights.csv", index=False)


def show_cost(cost):
    for index, house in enumerate(utils.HOUSES):
        plt.plot([i for i in range(len(cost[house]))], list(cost[house]),
                 c=utils.COLOR[index], label=house)
    plt.legend()
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_train_file", help="Please add a csv file as a parameter (.csv file)", type=str)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.1,
                        help="Choose the learning rate.\n")
    parser.add_argument("-e", "--stop_param", type=float, default=0.0001,
                        help="Choose the epsilon when iterations should stop.\n")
    parser.add_argument("-r", "--reg_param", type=float, default=100,
                        help="Choose the regularization parameter.\n")
    parser.add_argument("-s", "--split", type=float, default=0.7,
                        help="Choose the train_size split.\n")
    parser.add_argument("-v", "--visualisation", action="store_true",
                        help="Display cost function.\n")
    parser.add_argument("-a", "--accuracy", action="store_true",
                        help="Display train accuracy")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    train_data = pd.read_csv(args.dataset_train_file)

    final_theta_dict, cost_list_dict, train_accuracy = multiclass_classifier.train(
        train_data,
        alpha=args.learning_rate,
        epsilon=args.stop_param,
        reg_param=args.reg_param,
        train_size=args.split)

    theta_dict_to_csv(final_theta_dict)

    if args.accuracy:
        print("Train accuracy is: ", train_accuracy)
    if args.visualisation:
        show_cost(cost_list_dict)
