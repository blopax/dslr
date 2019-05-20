import pandas as pd
import matplotlib.pyplot as plt

import utils


def show_all(df):
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
    fig.savefig("histogram.pdf", bbox_inches='tight')


def hist(df):
    for i, house in enumerate(utils.HOUSES):
        plt.hist(df[df['Hogwarts House'] == house]["Care of Magical Creatures"], color=utils.COLOR[i], alpha=0.5)
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv("dataset_train.csv")
    # hist(train_df)
    show_all(train_df)
    # print(train_df[:5])
