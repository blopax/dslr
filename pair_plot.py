import pandas as pd
import matplotlib.pyplot as plt

SUBJECT = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
           "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures",
           "Charms", "Flying"]

HOUSES = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
COLOR = ["red", "blue", "green", "yellow"]

color_dict = {
    "Ravenclaw": "red",
    "Slytherin": "green",
    "Gryffindor": "yellow",
    "Hufflepuff": "blue"
}


def pair_plot(df):
    filtered_df = df.dropna()
    colors = filtered_df["Hogwarts House"].apply(lambda x: color_dict[x])
    filtered_df["house_nb"] = filtered_df["Hogwarts House"].astype('category').cat.codes
    filtered_df["hand"] = filtered_df["Best Hand"].astype('category').cat.codes
    filtered_df["date"] = pd.to_datetime(filtered_df["Birthday"])
    filtered_df["day"] = pd.to_datetime(filtered_df["Birthday"]).apply(lambda x: x.day)
    filtered_df["month"] = pd.to_datetime(filtered_df["Birthday"]).apply(lambda x: x.month)
    filtered_df.drop("Index", axis=1, inplace=True)
    pd.plotting.scatter_matrix(filtered_df, color=colors)
    plt.show()


if __name__ == "__main__":
    train_df = pd.read_csv("dataset_train.csv")
    pair_plot(train_df)
