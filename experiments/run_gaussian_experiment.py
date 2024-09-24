import datetime

import joblib
import numpy as np
from joblib import Parallel, delayed

import soft_label_learning.experiments.gaussians as gaussians
from soft_label_learning.config import path_output

start = 4  # Start point
end = 1e4  # End point

# Define the number of points
num_points = 50

# Generate a the points evenly spaced in log space
log_points = np.logspace(np.log10(start), np.log10(end), num=num_points)

# Convert log_points to integers and remove duplicates
n_samples_list = np.unique(log_points.astype(int))

# Different settings for the experiments
distance_values = [0.5]  # [0.1,0.25,0.5,1,2,5]
problem_dimensions = [2]  # [1, 2, 3, 5, 10, 50]
c1_probabilities = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
# noise_values = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5]

# For the two experiments in the paper, run experiments with noise = 0.0 and noise = 0.1
noise_values = [0.0]  # exp 1
# noise_values = [0.1]  # exp 2

settings_dict = {}
settings_dict["n_samples_list"] = n_samples_list
settings_dict["distance_values"] = distance_values
settings_dict["problem_dimensions"] = problem_dimensions
settings_dict["c1_probabilities"] = c1_probabilities
settings_dict["noise_values"] = noise_values

exp_settings = [
    (x, y, z, a)
    for x in distance_values
    for y in problem_dimensions
    for z in c1_probabilities
    for a in noise_values
]

repeats = 100000

results = Parallel(n_jobs=7, verbose=10)(
    delayed(gaussians.parallel_experiment)(n_samples_list, setting, repeats)
    for setting in exp_settings
)

exp_results = results[0].copy()
for i in range(1, len(results)):
    exp_results.update(results[i])


# Get current time, update paths
now = datetime.datetime.now()
time_string = now.strftime("%Y%m%d_%H_%M_%S")
result_string = time_string + "_results.joblib"
settings_string = time_string + "_settings.joblib"

# Save results and settings
save_path = path_output / "gaussians"
joblib.dump(exp_results, save_path / result_string)
joblib.dump(settings_dict, save_path / settings_string)
