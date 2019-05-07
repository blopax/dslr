import pandas as pd


def get_centile(serie, count, quantile_nb, quantile):
    quantile_position = int(quantile_nb / quantile * (count - 1))
    if ((count - 1) * quantile_nb / quantile) == int((count - 1) * quantile_nb / quantile):
        quantile_val = float(serie.iloc[quantile_position])
    else:
        quantile_val = ((quantile - quantile_nb) * float(serie.iloc[quantile_position])
                        + quantile_nb * float(serie.iloc[quantile_position + 1])) / quantile
    return quantile_val


def get_count_mean(serie):
    total, index = 0, 0
    for index, item in enumerate(serie):
        total += item
    count = float(index + 1)
    mean = total / count
    return count, mean


def get_std(serie, count, mean):
    deviation_sum = 0
    for item in serie:
        deviation_sum += ((float(item) - float(mean)) ** 2)
    std = (float(deviation_sum) / count) ** 0.5
    return std


def get_dispertion(serie, count):
    sorted_serie = serie.sort_values()
    minimum = float(sorted_serie.iloc[0])
    first_quartile = get_centile(sorted_serie, count, 1, 4)
    median = get_centile(sorted_serie, count, 1, 2)
    third_quartile = get_centile(sorted_serie, count, 3, 4)
    maximum = float(sorted_serie.iloc[int(count - 1)])
    return minimum, first_quartile, median, third_quartile, maximum


def describe_serie(serie):
    count, mean = get_count_mean(serie)
    std = get_std(serie, count, mean)
    minimum, first_quartile, median, third_quartile, maximum = get_dispertion(serie, count)
    return pd.Series([count, mean, std, minimum, first_quartile, median, third_quartile, maximum])


if __name__ == "__main__":
    df = pd.read_csv("dataset_train.csv")
    S = df["Arithmancy"].dropna()
    # S = pd.Series([0, 1, 2, 3 , 4 ,5 ,6 ,7])

    S_count, S_mean = get_count_mean(S)
    std_dev = get_std(S, S_count, S_mean)
    print(S_count, S_mean, std_dev)
    print(get_dispertion(S, S_count))
    print(S.describe())
