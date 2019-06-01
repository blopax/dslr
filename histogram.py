import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

import utils


def show_all(df):
    np.warnings.filterwarnings('ignore')
    fig = plt.figure(figsize=(10, 10))
    for index, subject in enumerate(utils.SUBJECT):
        ax = fig.add_subplot(len(utils.SUBJECT) / 4 + 1, 4, index + 1)
        for i, house in enumerate(utils.HOUSES):
            x_values = df[df['Hogwarts House'] == house][subject]
            ax.hist(x_values, color=utils.COLOR[i], alpha=0.5)
            ax.set_title(subject, fontsize=10)
            ax.set_xlabel('Grade', fontsize=8)
            ax.set_ylabel('Number of students', fontsize=8)
            ax.locator_params(nbins=5, tight=True)
            ax.tick_params(labelsize=5)

    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    fig.savefig("all_histogram.pdf", bbox_inches='tight')


def show_best(df):
    np.warnings.filterwarnings('ignore')
    for i, house in enumerate(utils.HOUSES):
        plt.hist(df[df['Hogwarts House'] == house]["Care of Magical Creatures"], color=utils.COLOR[i], alpha=0.5,
                 label=house)
        plt.legend()

    plt.xlabel('Grade')
    plt.ylabel('Number of students')
    plt.title("Repartition of grades between Poudlard's house")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("best_histogram.pdf", bbox_inches='tight')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--dataset_file", help="Please add a dataset file (.csv) as an argument.", type=str,
                        default='files/dataset_train.csv')
    parser.add_argument("-a", "--all", action="store_true",
                        help="show all histograms.\n")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # noinspection PyUnresolvedReferences
    try:
        data = pd.read_csv(args.dataset_file)
        if args.all:
            show_all(data)
            print("A pdf has been created in the folder.")
        else:
            show_best(data)
            print("A pdf has been created in the folder.")
    except FileNotFoundError as err:
        print("Error: {}".format(err))
    except pd.errors.ParserError as err:
        print("Error: dataset_train not csv or well formatted.\n{}".format(err))
    except UnicodeDecodeError as err:
        print("Error: dataset_train not csv or well formatted.\n{}".format(err))
