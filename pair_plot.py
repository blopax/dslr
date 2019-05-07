import pandas as pd
from pandas import plotting
import matplotlib.pyplot as plt

import utils


def pair_plot(df):
    filtered_df = df.dropna()
    colors = filtered_df["Hogwarts House"].apply(lambda x: utils.COLOR_DICT[x])
    filtered_df["house_nb"] = filtered_df["Hogwarts House"].astype('category').cat.codes
    filtered_df["hand"] = filtered_df["Best Hand"].astype('category').cat.codes
    filtered_df["date"] = pd.to_datetime(filtered_df["Birthday"])
    filtered_df["day"] = pd.to_datetime(filtered_df["Birthday"]).apply(lambda x: x.day)
    filtered_df["month"] = pd.to_datetime(filtered_df["Birthday"]).apply(lambda x: x.month)
    filtered_df.drop("Index", axis=1, inplace=True)
    plotting.scatter_matrix(filtered_df, color=colors)
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv("dataset_train.csv")
    pair_plot(train_df)