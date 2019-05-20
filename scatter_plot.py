import pandas as pd
import matplotlib.pyplot as plt

import utils


def scatter_plot_all(df):
    fig = plt.figure()
    i = 0
    for index1, subject1 in enumerate(utils.SUBJECT[:-1]):
        for index2, subject2 in enumerate(utils.SUBJECT[index1+1:]):
            i += 1
            ax = fig.add_subplot(len(utils.SUBJECT) * (len(utils.SUBJECT) + 1) / 2 / 10 + 1, 10, i)
            ax.plot(df[subject1], df[subject2])
    plt.show()


def scatter_plot(df):
    plt.scatter(df["Astronomy"], df["Defense Against the Dark Arts"])
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv(utils.TRAIN_FILE)
    scatter_plot_all(train_df)
    # scatter_plot(train_df)
