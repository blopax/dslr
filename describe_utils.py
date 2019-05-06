import pandas


def get_count_mean(serie):
    sum = 0
    for index, item in enumerate(serie):
        sum += item
    count = float(index + 1)
    mean = sum / count
    return count, mean


def get_std(serie, count, mean):
    deviation_sum = 0
    for item in serie:
        deviation_sum += (item - mean) ** 2
    std = (float(deviation_sum) / count) ** 0.5
    return std


def get_dispertion(serie, count):
    sorted_serie = serie.sort_values()
    min = float(sorted_serie[0])
    first_quartile = float(sorted_serie[int(count / 4)])
    median = float(sorted_serie[int(count/2)])
    third_quartile = float(sorted_serie[int(3 * count / 4)])
    max = float(sorted_serie[count - 1])
    return min, first_quartile, median, third_quartile, max


if __name__ == "__main__":
    df = pandas.read_csv("dataset_train.csv")
    S = df["Arithmancy"]
    count, mean = get_count_mean(S)
    std = get_std(S, count, mean)
    print(count, mean, std)
    print(get_dispertion(S, count))
