import logging

from soft_label_learning.config import path_repository
from soft_label_learning.experiments.synthetic_data import determine_mtvd_per_dataset

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(path_repository / "experiments" / "logs" / "debug_lr.log"),
        logging.StreamHandler(),
    ],
)

save_data = True
model_to_use = "lr"
fh_samples = 500

mtvd_dict = determine_mtvd_per_dataset(model_to_use, fh_samples, save_data)
