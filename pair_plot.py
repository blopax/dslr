import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
import utils
import argparse


def pair_plot(df):
    pd.set_option('mode.chained_assignment', None)
    filtered_df = df.dropna()
    colors = filtered_df["Hogwarts House"].apply(lambda x: utils.COLOR_DICT[x])
    filtered_df["house_nb"] = filtered_df["Hogwarts House"].astype('category').cat.codes
    filtered_df.drop("Index", axis=1, inplace=True)
    filtered_df = filtered_df[utils.SELECTED_SUBJECT]
    scatter_matrix = plotting.scatter_matrix(filtered_df, marker=".", s=1, diagonal="kde", color=colors)
    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=4, rotation=0)
        ax.set_ylabel(ax.get_ylabel(), fontsize=4, rotation=90)
        ax.tick_params(width=0, length=0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.savefig("scatter_matrix.pdf", bbox_inches='tight')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file", help="Please add a dataset file (.csv) as an argument.", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # noinspection PyUnresolvedReferences
    try:
        data = pd.read_csv(args.dataset_file)
        pair_plot(data)
        print("A pdf has been created in the folder.")
    except FileNotFoundError as err:
        print("Error: {}".format(err))
    except pd.errors.ParserError as err:
        print("Error: dataset_train not csv or well formatted.\n{}".format(err))
