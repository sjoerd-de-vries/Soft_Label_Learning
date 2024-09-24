import datetime
import logging

from soft_label_learning.config import path_repository
from soft_label_learning.experiments.experiment_settings import (
    q2_gt_generation_settings,
)
from soft_label_learning.experiments.synthetic_data import complete_experiment

# get current time
now = datetime.datetime.now()
time_string = now.strftime("%Y%m%d_%H_%M_%S")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            path_repository
            / "experiments"
            / "logs"
            / f"{time_string}_debug_complete_exp.log"
        ),
        logging.StreamHandler(),
    ],
)

# TODO adjust feature hiding paths
rf_feature_hiding_path = (
    path_repository
    / "experiments"
    / "feature_hiding_mtvd"
    / "date_hh_mm_ss_features_to_hide_rf.json"
)
lr_feature_hiding_path = (
    path_repository
    / "experiments"
    / "feature_hiding_mtvd"
    / "date_hh_mm_ss_features_to_hide_lr.json"
)

complete_experiment(
    q2_gt_generation_settings,
    lr_feature_hiding_path,
    rf_feature_hiding_path,
    fh_samples=1000,
    repeats=1,
    n_jobs=1,
    save=True,
    test_run=False,
    gt_save_and_load=True,
    generate_gt_sets=True,
)
