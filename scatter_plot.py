import pandas as pd
import matplotlib.pyplot as plt

SUBJECT = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
           "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures",
           "Charms", "Flying"]

HOUSES = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
COLOR = ["red", "blue", "green", "yellow"]


def scatter_plot_all(df):
    fig = plt.figure()
    i = 0
    for index1, subject1 in enumerate(SUBJECT[:-1]):
        for index2, subject2 in enumerate(SUBJECT[index1+1:]):
            i += 1
            ax = fig.add_subplot(len(SUBJECT) * (len(SUBJECT) + 1) / 2 / 10 + 1, 10, i)
            ax.plot(df[subject1], df[subject2])
    plt.show()


def scatter_plot(df):
    plt.scatter(df["Astronomy"], df["Defense Against the Dark Arts"])
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv("dataset_train.csv")
    scatter_plot_all(train_df)
    scatter_plot(train_df)
    # print(train_df[:5])
