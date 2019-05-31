import pandas as pd


def get_accuracy(prediction, truth):
    errors_loc = truth["Hogwarts House"] != prediction
    errors = pd.DataFrame()
    errors["truth"] = truth[errors_loc]["Hogwarts House"]
    errors["prediction"] = prediction[errors_loc]

    total_accuracy = 1.0 - len(errors) / len(truth)
    house_acc = dict()
    for house in HOUSES:
        house_acc[house] = 1 - len(errors[errors["truth"] == house]) / len(truth[truth["Hogwarts House"] == house])

    accuracy_dict = dict()
    accuracy_dict['total'] = total_accuracy
    accuracy_dict['errors'] = errors
    accuracy_dict['house accuracy'] = [house_acc]
    return accuracy_dict


SUBJECT = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
           "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures",
           "Charms", "Flying"]
SELECTED_SUBJECT = ["Astronomy", "Herbology", "Divination", "Muggle Studies", "Ancient Runes", "Transfiguration",
                    "Charms", "Flying"]

SELECTED_FEATURES = ["Charms", "Astronomy", "Flying"]

HOUSES = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]

COLOR = ["red", "green", "yellow", "blue"]
COLOR_DICT = {
    "Ravenclaw": "red",
    "Slytherin": "green",
    "Gryffindor": "yellow",
    "Hufflepuff": "blue"
}


OUTPUT_COLUMN = "Hogwarts House"
Y_COLUMN = "house_class"

TRAIN_FILE = "files/dataset_train.csv"
TEST_FILE = "files/dataset_test.csv"
