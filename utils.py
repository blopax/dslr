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

SELECTED_FEATURES =["Herbology", "Ancient Runes", "Flying"]

COMBINATORY = [['Astronomy', 'Herbology', 'Divination', 'Muggle Studies'], ['Astronomy', 'Herbology', 'Divination', 'Ancient Runes'], ['Astronomy', 'Herbology', 'Divination', 'Transfiguration'], ['Astronomy', 'Herbology', 'Divination', 'Charms'], ['Astronomy', 'Herbology', 'Divination', 'Flying'], ['Astronomy', 'Herbology', 'Muggle Studies', 'Ancient Runes'], ['Astronomy', 'Herbology', 'Muggle Studies', 'Transfiguration'], ['Astronomy', 'Herbology', 'Muggle Studies', 'Charms'], ['Astronomy', 'Herbology', 'Muggle Studies', 'Flying'], ['Astronomy', 'Herbology', 'Ancient Runes', 'Transfiguration'], ['Astronomy', 'Herbology', 'Ancient Runes', 'Charms'], ['Astronomy', 'Herbology', 'Ancient Runes', 'Flying'], ['Astronomy', 'Herbology', 'Transfiguration', 'Charms'], ['Astronomy', 'Herbology', 'Transfiguration', 'Flying'], ['Astronomy', 'Herbology', 'Charms', 'Flying'], ['Astronomy', 'Divination', 'Muggle Studies', 'Ancient Runes'], ['Astronomy', 'Divination', 'Muggle Studies', 'Transfiguration'], ['Astronomy', 'Divination', 'Muggle Studies', 'Charms'], ['Astronomy', 'Divination', 'Muggle Studies', 'Flying'], ['Astronomy', 'Divination', 'Ancient Runes', 'Transfiguration'], ['Astronomy', 'Divination', 'Ancient Runes', 'Charms'], ['Astronomy', 'Divination', 'Ancient Runes', 'Flying'], ['Astronomy', 'Divination', 'Transfiguration', 'Charms'], ['Astronomy', 'Divination', 'Transfiguration', 'Flying'], ['Astronomy', 'Divination', 'Charms', 'Flying'], ['Astronomy', 'Muggle Studies', 'Ancient Runes', 'Transfiguration'], ['Astronomy', 'Muggle Studies', 'Ancient Runes', 'Charms'], ['Astronomy', 'Muggle Studies', 'Ancient Runes', 'Flying'], ['Astronomy', 'Muggle Studies', 'Transfiguration', 'Charms'], ['Astronomy', 'Muggle Studies', 'Transfiguration', 'Flying'], ['Astronomy', 'Muggle Studies', 'Charms', 'Flying'], ['Astronomy', 'Ancient Runes', 'Transfiguration', 'Charms'], ['Astronomy', 'Ancient Runes', 'Transfiguration', 'Flying'], ['Astronomy', 'Ancient Runes', 'Charms', 'Flying'], ['Astronomy', 'Transfiguration', 'Charms', 'Flying'], ['Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes'], ['Herbology', 'Divination', 'Muggle Studies', 'Transfiguration'], ['Herbology', 'Divination', 'Muggle Studies', 'Charms'], ['Herbology', 'Divination', 'Muggle Studies', 'Flying'], ['Herbology', 'Divination', 'Ancient Runes', 'Transfiguration'], ['Herbology', 'Divination', 'Ancient Runes', 'Charms'], ['Herbology', 'Divination', 'Ancient Runes', 'Flying'], ['Herbology', 'Divination', 'Transfiguration', 'Charms'], ['Herbology', 'Divination', 'Transfiguration', 'Flying'], ['Herbology', 'Divination', 'Charms', 'Flying'], ['Herbology', 'Muggle Studies', 'Ancient Runes', 'Transfiguration'], ['Herbology', 'Muggle Studies', 'Ancient Runes', 'Charms'], ['Herbology', 'Muggle Studies', 'Ancient Runes', 'Flying'], ['Herbology', 'Muggle Studies', 'Transfiguration', 'Charms'], ['Herbology', 'Muggle Studies', 'Transfiguration', 'Flying'], ['Herbology', 'Muggle Studies', 'Charms', 'Flying'], ['Herbology', 'Ancient Runes', 'Transfiguration', 'Charms'], ['Herbology', 'Ancient Runes', 'Transfiguration', 'Flying'], ['Herbology', 'Ancient Runes', 'Charms', 'Flying'], ['Herbology', 'Transfiguration', 'Charms', 'Flying'], ['Divination', 'Muggle Studies', 'Ancient Runes', 'Transfiguration'], ['Divination', 'Muggle Studies', 'Ancient Runes', 'Charms'], ['Divination', 'Muggle Studies', 'Ancient Runes', 'Flying'], ['Divination', 'Muggle Studies', 'Transfiguration', 'Charms'], ['Divination', 'Muggle Studies', 'Transfiguration', 'Flying'], ['Divination', 'Muggle Studies', 'Charms', 'Flying'], ['Divination', 'Ancient Runes', 'Transfiguration', 'Charms'], ['Divination', 'Ancient Runes', 'Transfiguration', 'Flying'], ['Divination', 'Ancient Runes', 'Charms', 'Flying'], ['Divination', 'Transfiguration', 'Charms', 'Flying'], ['Muggle Studies', 'Ancient Runes', 'Transfiguration', 'Charms'], ['Muggle Studies', 'Ancient Runes', 'Transfiguration', 'Flying'], ['Muggle Studies', 'Ancient Runes', 'Charms', 'Flying'], ['Muggle Studies', 'Transfiguration', 'Charms', 'Flying'], ['Ancient Runes', 'Transfiguration', 'Charms', 'Flying'], ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes'], ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Transfiguration'], ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Charms'], ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Flying'], ['Astronomy', 'Herbology', 'Divination', 'Ancient Runes', 'Transfiguration'], ['Astronomy', 'Herbology', 'Divination', 'Ancient Runes', 'Charms'], ['Astronomy', 'Herbology', 'Divination', 'Ancient Runes', 'Flying'], ['Astronomy', 'Herbology', 'Divination', 'Transfiguration', 'Charms'], ['Astronomy', 'Herbology', 'Divination', 'Transfiguration', 'Flying'], ['Astronomy', 'Herbology', 'Divination', 'Charms', 'Flying'], ['Astronomy', 'Herbology', 'Muggle Studies', 'Ancient Runes', 'Transfiguration'], ['Astronomy', 'Herbology', 'Muggle Studies', 'Ancient Runes', 'Charms'], ['Astronomy', 'Herbology', 'Muggle Studies', 'Ancient Runes', 'Flying'], ['Astronomy', 'Herbology', 'Muggle Studies', 'Transfiguration', 'Charms'], ['Astronomy', 'Herbology', 'Muggle Studies', 'Transfiguration', 'Flying'], ['Astronomy', 'Herbology', 'Muggle Studies', 'Charms', 'Flying'], ['Astronomy', 'Herbology', 'Ancient Runes', 'Transfiguration', 'Charms'], ['Astronomy', 'Herbology', 'Ancient Runes', 'Transfiguration', 'Flying'], ['Astronomy', 'Herbology', 'Ancient Runes', 'Charms', 'Flying'], ['Astronomy', 'Herbology', 'Transfiguration', 'Charms', 'Flying'], ['Astronomy', 'Divination', 'Muggle Studies', 'Ancient Runes', 'Transfiguration'], ['Astronomy', 'Divination', 'Muggle Studies', 'Ancient Runes', 'Charms'], ['Astronomy', 'Divination', 'Muggle Studies', 'Ancient Runes', 'Flying'], ['Astronomy', 'Divination', 'Muggle Studies', 'Transfiguration', 'Charms'], ['Astronomy', 'Divination', 'Muggle Studies', 'Transfiguration', 'Flying'], ['Astronomy', 'Divination', 'Muggle Studies', 'Charms', 'Flying'], ['Astronomy', 'Divination', 'Ancient Runes', 'Transfiguration', 'Charms'], ['Astronomy', 'Divination', 'Ancient Runes', 'Transfiguration', 'Flying'], ['Astronomy', 'Divination', 'Ancient Runes', 'Charms', 'Flying'], ['Astronomy', 'Divination', 'Transfiguration', 'Charms', 'Flying'], ['Astronomy', 'Muggle Studies', 'Ancient Runes', 'Transfiguration', 'Charms'], ['Astronomy', 'Muggle Studies', 'Ancient Runes', 'Transfiguration', 'Flying'], ['Astronomy', 'Muggle Studies', 'Ancient Runes', 'Charms', 'Flying'], ['Astronomy', 'Muggle Studies', 'Transfiguration', 'Charms', 'Flying'], ['Astronomy', 'Ancient Runes', 'Transfiguration', 'Charms', 'Flying'], ['Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes', 'Transfiguration'], ['Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes', 'Charms'], ['Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes', 'Flying'], ['Herbology', 'Divination', 'Muggle Studies', 'Transfiguration', 'Charms'], ['Herbology', 'Divination', 'Muggle Studies', 'Transfiguration', 'Flying'], ['Herbology', 'Divination', 'Muggle Studies', 'Charms', 'Flying'], ['Herbology', 'Divination', 'Ancient Runes', 'Transfiguration', 'Charms'], ['Herbology', 'Divination', 'Ancient Runes', 'Transfiguration', 'Flying'], ['Herbology', 'Divination', 'Ancient Runes', 'Charms', 'Flying'], ['Herbology', 'Divination', 'Transfiguration', 'Charms', 'Flying'], ['Herbology', 'Muggle Studies', 'Ancient Runes', 'Transfiguration', 'Charms'], ['Herbology', 'Muggle Studies', 'Ancient Runes', 'Transfiguration', 'Flying'], ['Herbology', 'Muggle Studies', 'Ancient Runes', 'Charms', 'Flying'], ['Herbology', 'Muggle Studies', 'Transfiguration', 'Charms', 'Flying'], ['Herbology', 'Ancient Runes', 'Transfiguration', 'Charms', 'Flying'], ['Divination', 'Muggle Studies', 'Ancient Runes', 'Transfiguration', 'Charms'], ['Divination', 'Muggle Studies', 'Ancient Runes', 'Transfiguration', 'Flying'], ['Divination', 'Muggle Studies', 'Ancient Runes', 'Charms', 'Flying'], ['Divination', 'Muggle Studies', 'Transfiguration', 'Charms', 'Flying'], ['Divination', 'Ancient Runes', 'Transfiguration', 'Charms', 'Flying'], ['Muggle Studies', 'Ancient Runes', 'Transfiguration', 'Charms', 'Flying']]

# SELECTED_FEATURES = ["Charms", "Divination", "Flying"]  # 0.9725ac reg a 100

OUTPUT_COLUMN = "Hogwarts House"
Y_COLUMN = "house_class"

TRAIN_FILE = "dataset_train.csv"
TEST_FILE = "dataset_test.csv"
