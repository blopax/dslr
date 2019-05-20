import itertools

from sources import utils

lst = utils.SUBJECT
combs = []

for i in range(1, len(lst)+1):
    combs.append(i)
    els = [list(x) for x in itertools.combinations(lst, i)]
    combs.append(els)

combs_cleaned = []
for each in combs:
    if type(each) == list:
        for combinatory in each:
            if 1 < len(combinatory) < 4:
                combs_cleaned.append(combinatory)

print(combs_cleaned)
