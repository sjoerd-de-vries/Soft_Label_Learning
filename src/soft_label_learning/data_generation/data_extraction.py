import pandas as pd

from ..config import path_uci_keel_data

datasets = [
    "keel_car.csv",
    "keel_contraceptive.csv",
    "keel_flare.csv",
    "keel_pima.csv",
    "keel_spambase.csv",
    "keel_titanic.csv",
    "keel_vehicle.csv",
    "keel_vowel.csv",
    "uci_abalone.csv",
    "uci_australian.csv",
    "uci_german.csv",
    "uci_madelon.csv",
    "uci_mice.csv",
    "uci_nursery.csv",
    "uci_red.csv",
    "uci_white.csv",
    "uci_yeast.csv",
]

binary_datasets = [
    "keel_pima.csv",
    "keel_spambase.csv",
    "keel_titanic.csv",
    "uci_australian.csv",
    "uci_german.csv",
    "uci_red.csv",
    "uci_white.csv",
    "uci_madelon.csv",
]

multiclass_datasets = [
    "keel_car.csv",
    "keel_contraceptive.csv",
    "keel_flare.csv",
    "keel_vehicle.csv",
    "keel_vowel.csv",
    "uci_abalone.csv",
    "uci_nursery.csv",
    "uci_yeast.csv",
    "uci_mice.csv",
]


def read_data(
    dataset_name,
    read_path=None,
):
    if read_path is None:
        read_path = path_uci_keel_data

    dataset = pd.read_csv(read_path / dataset_name, sep=";", header=None)
    dataset = dataset.to_numpy()

    if dataset_name == "uci_nursery.csv":
        # exclude rows where the last column is 2, as there are only two of those
        dataset = dataset[dataset[:, -1] != 2]
        # rename the columns
        dataset[dataset[:, -1] == 3] = 2
        dataset[dataset[:, -1] == 4] = 3

    return dataset
