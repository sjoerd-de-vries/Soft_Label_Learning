import datetime
import json

from soft_label_learning.config import path_repository
from soft_label_learning.experiments.experiment_settings import (
    methods_without_threshold,
    non_ens_methods,
    q2_settings,
)
from soft_label_learning.experiments.process_synthetic_data import get_q2_result_dict

# TODO set datetime to the desired result folder
settings_dict, result_path = q2_settings, "date_hh_mm_ss"

methods = methods_without_threshold

datasets = [x[:-4] for x in settings_dict["dataset"]]
classifiers = [x for x in settings_dict["clf"]]
metrics = ["TVD", "hard_soft_AUC"]
fixed_settings = settings_dict.copy()
fixed_settings["ens_propagation"] = True
fixed_settings["alpha"] = 1

# set path for loading and saving results
save_path = path_repository / "experiments" / "synthetic_data_results"

result_dict = get_q2_result_dict(
    fixed_settings,
    methods,
    metrics,
    classifiers,
    datasets,
    non_ens_methods,
    result_path,
)

result_dict_converted = json.loads(json.dumps(result_dict))

# save dict to file
now = datetime.datetime.now()
time_string = now.strftime("%Y%m%d_%H_%M_%S")

with open(save_path / f"q2_results_processed_{time_string}.json", "w") as f:
    json.dump(result_dict_converted, f)
