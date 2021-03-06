import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

import utils


def scatter_plot_clustered(df):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    i = 0
    for index1, subject1 in enumerate(utils.SUBJECT[:-1]):
        for subject2 in utils.SUBJECT[index1 + 1:]:
            i += 1
            ax = fig.add_subplot(len(utils.SUBJECT) * (len(utils.SUBJECT) + 1) / 2 / 12 + 1, 12, i)
            ax.scatter(df[subject1], df[subject2], marker='.')
            ax.set_title(subject1[:8] + ' - ' + subject2[:8], fontsize=8)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.tick_params(labelsize=2)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    fig.savefig("scatter_plot.png", dpi=100)


def scatter_plot_detailed(df):
    pp = PdfPages('scatter_plot_detailed.pdf')
    i = 0
    for index1, subject1 in enumerate(utils.SUBJECT[:-1]):
        for subject2 in utils.SUBJECT[index1 + 1:]:
            i += 1
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.scatter(df[subject1], df[subject2], marker='o')
            ax.set_title(subject1 + ' - ' + subject2, fontsize=14)
            ax.set_xlabel(subject1, fontsize=10)
            ax.set_ylabel(subject2, fontsize=10)
            ax.tick_params(labelsize=8)
            pp.savefig()
            plt.close()
    pp.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--dataset_file", help="Please add a dataset file (.csv) as an argument.", type=str,
                        default='files/dataset_train.csv')
    parser.add_argument("-d", "--detailed", action="store_true",
                        help="show all scatter plot on different page with more details.\n")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # noinspection PyUnresolvedReferences
    try:
        data = pd.read_csv(args.dataset_file)
        if args.detailed:
            scatter_plot_detailed(data)
            print("A pdf has been created in the folder.")
        else:
            scatter_plot_clustered(data)
            print("A png has been created in the folder.")
    except FileNotFoundError as err:
        print("Error: {}".format(err))
    except pd.errors.ParserError as err:
        print("Error: dataset_train not csv or well formatted.\n{}".format(err))
    except UnicodeDecodeError as err:
        print("Error: dataset_train not csv or well formatted.\n{}".format(err))
