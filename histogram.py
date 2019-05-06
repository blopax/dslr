import pandas as pd
import matplotlib.pyplot as plt

SUBJECT = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
           "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures",
           "Charms", "Flying"]

HOUSES = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
COLOR = ["red", "blue", "green", "yellow"]


def show_all(df):
    fig = plt.figure()
    ax_list = []
    for index, subject in enumerate(SUBJECT):
        ax = fig.add_subplot(len(SUBJECT) / 4 + 1, 4, index + 1)
        for i, house in enumerate(HOUSES):
            x_values = df[df['Hogwarts House'] == house][subject]
            ax.hist(x_values, color=COLOR[i], alpha=0.5)
            ax_list.append(ax)
    plt.show()


def hist(df):
    for i, house in enumerate(HOUSES):
        plt.hist(df[df['Hogwarts House'] == house]["Care of Magical Creatures"], color=COLOR[i], alpha = 0.5)
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv("dataset_train.csv")
    hist(train_df)
    # show_all(train_df)
    # print(train_df[:5])
