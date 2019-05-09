import itertools

lst = ["Astronomy", "Herbology", "Divination", "Muggle Studies",
           "Ancient Runes", "Transfiguration", "Charms", "Flying"]
combs = []

for i in range(1, len(lst)+1):
    combs.append(i)
    els = [list(x) for x in itertools.combinations(lst, i)]
    combs.append(els)

combs_cleaned = []
for each in combs:
    if type(each) == list:
        for combinatory in each:
            if len(combinatory) > 3 and len(combinatory) < 6:
                combs_cleaned.append(combinatory)

print(combs_cleaned)