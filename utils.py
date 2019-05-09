SUBJECT = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
           "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures",
           "Charms", "Flying"]

HOUSES = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
COLOR = ["red", "green", "yellow", "blue"]


COLOR_DICT = {
    "Ravenclaw": "red",
    "Slytherin": "green",
    "Gryffindor": "yellow",
    "Hufflepuff": "blue"
}

SELECTED_SUBJECT = ["Astronomy", "Herbology", "Divination", "Muggle Studies",
           "Ancient Runes", "Transfiguration", "Charms", "Flying"]

# SELECTED_FEATURES = ["Herbology", "Ancient Runes"]
# SELECTED_FEATURES = ["Muggle Studies", "Astronomy", "Flying"]
#SELECTED_FEATURES = ["Ancient Runes", "Astronomy", "Herbology"]
# SELECTED_FEATURES = ["Astronomy", "Herbology", "Divination", "Muggle Studies", "Ancient Runes", "Transfiguration", "Charms", "Flying"]
#SELECTED_FEATURES = ["Herbology", "Astronomy", "Charms", "Muggle Studies"]

SELECTED_FEATURES =["Divination", "Charms", "Flying", "Ancient Runes"]
# SELECTED_FEATURES = ["Astronomy", "Herbology", "Divination", "Muggle Studies", "Ancient Runes", "Transfiguration", "Charms", "Flying"]

# SELECTED_FEATURES = ["Charms", "Divination", "Flying"]  # 0.9725ac reg a 100

OUTPUT_COLUMN = "Hogwarts House"
Y_COLUMN = "house_class"

TRAIN_FILE = "dataset_train.csv"
TEST_FILE = "dataset_test.csv"
