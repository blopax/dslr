import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt
import utils
import scipy

def pair_plot(df):
    filtered_df = df.dropna()
    colors = filtered_df["Hogwarts House"].apply(lambda x: utils.COLOR_DICT[x])
    filtered_df.loc["house_nb"] = filtered_df["Hogwarts House"].astype('category').cat.codes
    filtered_df.drop("Index", axis=1, inplace=True)
    filtered_df = filtered_df[utils.SELECTED_SUBJECT]
    scatter_matrix = plotting.scatter_matrix(filtered_df, marker= ".", s=1, diagonal = "kde", color=colors)
    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize = 4, rotation = 0)
        ax.set_ylabel(ax.get_ylabel(), fontsize = 4, rotation = 90)
        ax.tick_params(width=0, length=0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    plt.savefig("scatter_matrix.pdf", bbox_inches='tight')


def test_plot(df):
    filtered_df = df.dropna()
    # colors = filtered_df["Hogwarts House"].apply(lambda x: utils.COLOR_DICT[x])
    filtered_df["house_nb"] = filtered_df["Hogwarts House"].astype('category').cat.codes
    filtered_df["hand"] = filtered_df["Best Hand"].astype('category').cat.codes
    filtered_df["date"] = pd.to_datetime(filtered_df["Birthday"])
    filtered_df["day"] = pd.to_datetime(filtered_df["Birthday"]).apply(lambda x: x.day)
    filtered_df["month"] = pd.to_datetime(filtered_df["Birthday"]).apply(lambda x: x.month)
    # filtered_df.drop("Index", axis=1, inplace=True)
    # filtered_df = filtered_df[["house_nb", "hand", "day", "month"]]
    # plotting.scatter_matrix(filtered_df, color=colors)
    # plt.show()

    for i, house in enumerate(utils.HOUSES):
        # plt.hist(filtered_df[filtered_df['Hogwarts House'] == house]["hand"], color=utils.COLOR[i], alpha=0.5)
        # plt.hist(filtered_df[filtered_df['Hogwarts House'] == house]["month"], color=utils.COLOR[i], alpha=0.5)
        # plt.hist(filtered_df[filtered_df['Hogwarts House'] == house]["day"], color=utils.COLOR[i], alpha=0.5)
        plt.hist(filtered_df[filtered_df['Hogwarts House'] == house]["date"], color=utils.COLOR[i], alpha=0.5)
        plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv("dataset_train.csv")
    pair_plot(train_df)
    # test_plot(train_df)
