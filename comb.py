import itertools

lst = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
           "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures",
           "Charms", "Flying"]
combs = []

for i in range(1, len(lst)+1):
    combs.append(i)
    els = [list(x) for x in itertools.combinations(lst, i)]
    combs.append(els)

combs_cleaned = []
for each in combs:
    if type(each) == list:
        for combinatory in each:
            if len(combinatory) > 1 and len(combinatory) < 4:
                combs_cleaned.append(combinatory)

print(combs_cleaned)