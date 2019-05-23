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
    parser.add_argument("-l", "--learning_rate", type=float, default=0.005,
                        help="Choose the learning rate.\n")
    parser.add_argument("-e", "--epsilon", type=float, default=0.0001,
                        help="Choose the epsilon when iterations should stop.\n")
    parser.add_argument("-r", "--reg_param", type=float, default=None,
                        help="Choose the regularization parameter.\n")
    parser.add_argument("-s", "--split", type=float, default=0.7,
                        help="Choose the train_size split.\n")
    parser.add_argument("-v", "--visualisation", action="store_true",
                        help="Display cost function.\n")
    parser.add_argument("-a", "--accuracy", type=str, default=None, choices=['simple', 'full'],
                        help="Display train accuracy if simple and more information if full")
    parser.add_argument("-m", "--mode", type=str, default="batch", choices=["batch", "mini_batch", "stochastic"],
                        help="Choose gradient descent mode.")
    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="For mini_batch. Pick the mini_batch size.\n")
    parser.add_argument("-i", "--iterations", type=int, default=None,
                        help="For mini_batch and stochastic. Pick the number of iterations.\n")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if not 0 < args.split < 1:
        print("Train split must belong to ]0, 1[")
        exit(0)
    # noinspection PyUnresolvedReferences
    try:
        train_data = pd.read_csv(args.dataset_train_file)
        if not args.reg_param and args.mode == "batch":
            args.reg_param = 100
        elif not args.reg_param:
            args.reg_param = 0
        if args.mode == "stochastic":
            args.batch_size = 1
        if not args.iterations and args.mode == "stochastic":
            args.iterations = 5
        if not args.batch_size and args.mode == "mini_batch":
            args.batch_size = 50
        if not args.iterations and args.mode == "mini_batch":
            args.iterations = 75
        if args.mode == "mini_batch" and not 0 < args.batch_size <= len(train_data) * args.split:
            print("Mini batch must be strictly positive and less than len(train_data) * split.")
            exit(0)
        if args.iterations is not None and args.iterations < 1:
            print("Iterations must be strictly positive.")
            exit(0)

        final_theta_dict, cost_list_dict, accuracy_dict = multiclass_classifier.train(
            train_data,
            alpha=args.learning_rate,
            epsilon=args.epsilon,
            reg_param=args.reg_param,
            train_size=args.split,
            mode=args.mode,
            batch_size=args.batch_size,
            iterations=args.iterations)

        theta_dict_to_csv(final_theta_dict)

        if args.accuracy == 'full':
            print("Train total accuracy is: {}\n".format(accuracy_dict['total']))
            print("Accuracy per house is:\n{}\n".format(
                pd.DataFrame(accuracy_dict['house accuracy'], index=['accuracy'])))
            print("Wrong predictions are:\n{}".format(accuracy_dict['errors']))
        if args.accuracy == 'simple':
            print("Train total accuracy is: {}\n".format(accuracy_dict['total']))

        if args.visualisation:
            show_cost(cost_list_dict)
    except FileNotFoundError as err:
        print("Error: {}".format(err))
    except pd.errors.ParserError as err:
        print("Error: dataset_train not csv or well formatted.\n{}".format(err))
