import json
from datetime import datetime

from soft_label_learning.config import path_output
from soft_label_learning.data_generation.real_world_processing import get_processed_data
from soft_label_learning.experiments import real_world
from soft_label_learning.experiments.experiment_settings import real_world_settings

X, y_hard, y_soft = get_processed_data()

exp_settings = real_world_settings.copy()


## Settings
train_fractions = [5, 10, 20, 40, 60, 80]
alpha = 1


result_dict = {}

for train_fraction in train_fractions:
    print(train_fraction)
    results = real_world.run_experiment(
        exp_settings,
        X,
        y_soft,
        repeats=1000,
        n_jobs=60,
        save=False,
        alpha=alpha,
        train_percentage=train_fraction / 100.0,
    )

    result_dict[int(train_fraction)] = results

## save result_dict as json
# get current time
now = datetime.now()
time_string = now.strftime("%Y%m%d_%H_%M_%S")

complete_path = path_output / "real_world" / f"{time_string}_result_dict.json"

with open(complete_path, "w") as f:
    json.dump(result_dict, f)
