import pandas as pd


def get_quantile(serie, count, quantile_nb, quantile):
    if serie.empty or count <= 0 or quantile == 0:
        return None
    quantile_proportion = float(quantile_nb / quantile)
    float_position = (count - 1) * quantile_proportion
    floor = int(float_position)
    if float_position == floor:
        return serie.iloc[floor]
    ceil = floor + 1
    return serie.iloc[floor] * (ceil - float_position) + serie.iloc[ceil] * (float_position - floor)


def get_count_mean(serie):
    if serie.empty:
        return 0, None
    total, index = 0, 0
    for index, item in enumerate(serie):
        total += item
    count = float(index + 1)
    mean = total / count
    return count, mean


def get_std(serie, count, mean):
    if count <= 1:
        return None
    deviation_sum = 0.0
    i = 0
    for item in serie:
        deviation_sum += (float(item) - float(mean)) ** 2
        i += 1
    std = (float(deviation_sum) / (count - 1)) ** 0.5
    return std


def get_dispertion(serie, count):
    sorted_serie = serie.sort_values()
    if not serie.empty:
        minimum = float(sorted_serie.iloc[0])
        maximum = float(sorted_serie.iloc[int(count - 1)])
    else:
        minimum, maximum = None, None
    first_quartile = get_quantile(sorted_serie, count, 1, 4)
    median = get_quantile(sorted_serie, count, 1, 2)
    third_quartile = get_quantile(sorted_serie, count, 3, 4)
    return minimum, first_quartile, median, third_quartile, maximum


def describe_serie(serie):
    count, mean = get_count_mean(serie)
    std = get_std(serie, count, mean)
    minimum, first_quartile, median, third_quartile, maximum = get_dispertion(serie, count)
    return pd.Series([count, mean, std, minimum, first_quartile, median, third_quartile, maximum])


if __name__ == "__main__":
    df = pd.read_csv("dataset_train.csv")
    S = df["Arithmancy"].dropna()
    # S = pd.Series([])
    # S = pd.Series([1, 2, 3, 4])
    print(describe_serie(S))
    print(S.describe())
