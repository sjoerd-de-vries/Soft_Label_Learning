from ..data_generation.data_extraction import datasets

## All possible parameter values for the experiments

## Keys in order
dataset_keys = datasets
gt_keys = ["rf", "lr"]
mtvd_keys = ["1", "2"]
noise_keys = list(range(5, 35, 5))
noise_keys = [str(noise) for noise in noise_keys]
noise_type_keys = [
    "noiseless",
    "NCAR",
    "NAR",
    "miscalibrated_pos_false",
    "miscalibrated_neg_false",
    "miscalibrated_pos_true",
    "miscalibrated_neg_true",
]
clf_keys = ["LR", "SGD", "GNB", "DT"]

method_keys = [
    "PluralityClf",
    "PluralityWeightsClf",
    "ThresholdClf_25",
    "ThresholdClf_50",
    "ThresholdClf_75",
    "ThresholdWeightsClf_25",
    "ThresholdWeightsClf_50",
    "ThresholdWeightsClf_75",
    "SampleClf",
    "DuplicateWeightsClf",
    "PluralityBootstrapClf",
    "PluralityBootstrapWFClf",
    "PluralityEnsClf",
    "ThresholdBootstrapClf_25",
    "ThresholdBootstrapClf_50",
    "ThresholdBootstrapClf_75",
    "ThresholdBootstrapWFClf_25",
    "ThresholdBootstrapWFClf_50",
    "ThresholdBootstrapWFClf_75",
    "ThresholdEnsClf_25",
    "ThresholdEnsClf_50",
    "ThresholdEnsClf_75",
    "BootstrapSamplingClf",
    "EnsSamplingClf",
    "DupEnsClf",
    "BootstrapDupWeightsClf",
    "EnsDupWeightsClf",
    "BootstrapDupSamplingClf",
    "EnsDupSamplingClf",
]

methods_with_threshold = [x for x in method_keys if "Threshold" in x]
methods_without_threshold = [x for x in method_keys if x not in methods_with_threshold]

ens_methods = [x for x in method_keys if ("Ens" in x or "Bootstrap" in x)]
non_ens_methods = [x for x in method_keys if x not in ens_methods]

multi_label_methods = [
    x for x in method_keys if ("Dup" in x or "Sampling" in x or "_s" in x)
]
single_label_methods = [x for x in method_keys if x not in multi_label_methods]

refined_methods = [
    "PluralityClf",
    "PluralityWeightsClf",
    "SampleClf",
    "DuplicateWeightsClf",
    "PluralityEnsClf",
    "PluralityBootstrapClf",
    "PluralityBootstrapWFClf",
    "DupEnsClf",
    "BootstrapSamplingClf",
    "EnsSamplingClf",
    "BootstrapDupSamplingClf",
    "EnsDupSamplingClf",
    "BootstrapDupWeightsClf",
    "EnsDupWeightsClf",
]

# Repeats are not in the dict
label_eval_keys = ["hard", "soft", "hard-soft"]

# alpha is not used in the experiments
alpha_keys = [1]

# For label eval key hard
eval_set_keys = ["G", "PG_pv", "PG_s", "OH_pv", "OH_s"]
metric_keys = ["accuracy"]
# For label eval key soft
eval_set_keys = ["G_dist", "PG", "OS", "OH_pv_dist", "OH_s_dist"]
metric_keys = ["LL", "TVD", "MSE"]
# for label eval key hard-soft
eval_set_keys = ["G_dist", "OH_pv_dist", "OH_s_dist"]
metric_keys = ["hard_soft_LL", "hard_soft_AUC"]

train_test_keys = ["train", "test"]
ens_propagation_keys = True  # {True, False}

# Smoothing is not used in the experiments
# Could be anywhere from 0.0 to 0.x
smoothing_keys = 0.0

complete_settings_dict = {}
complete_settings_dict["dataset"] = dataset_keys
complete_settings_dict["gt"] = gt_keys
complete_settings_dict["mtvd"] = mtvd_keys
complete_settings_dict["noise"] = noise_keys
complete_settings_dict["noise_type"] = noise_type_keys
complete_settings_dict["clf"] = clf_keys
complete_settings_dict["method"] = method_keys
complete_settings_dict["label_eval"] = label_eval_keys
complete_settings_dict["alpha"] = alpha_keys
complete_settings_dict["eval_set"] = {}
complete_settings_dict["metric"] = {}

# setting the eval_set and metric for each label_eval key
for key in complete_settings_dict["label_eval"]:
    if key == "hard":
        complete_settings_dict["eval_set"][key] = [
            "G",
            "PG_pv",
            "PG_s",
            "OH_pv",
            "OH_s",
        ]
        complete_settings_dict["metric"][key] = ["accuracy"]
    elif key == "soft":
        complete_settings_dict["eval_set"][key] = [
            "G_dist",
            "PG",
            "OS",
            "OH_pv_dist",
            "OH_s_dist",
        ]
        complete_settings_dict["metric"][key] = ["LL", "TVD", "MSE"]
    elif key == "hard-soft":
        complete_settings_dict["eval_set"][key] = ["G_dist", "OH_pv_dist", "OH_s_dist"]
        complete_settings_dict["metric"][key] = ["hard_soft_LL", "hard_soft_AUC"]

complete_settings_dict["train_test"] = train_test_keys
complete_settings_dict["ens_propagation"] = ens_propagation_keys
complete_settings_dict["smoothing"] = smoothing_keys


# Q1: No noise
q1_settings = complete_settings_dict.copy()
q1_settings["noise"] = ["0"]
q1_settings["noise_type"] = ["noiseless"]
q1_settings["method"] = method_keys

# Q2: Noise
q2_settings = complete_settings_dict.copy()
q2_settings["gt"] = ["rf"]
q2_settings["method"] = refined_methods

# Generate D^G and D^PG in advance for the Q2 experiments
q2_gt_generation_settings = complete_settings_dict.copy()

# Real world data experiment
real_world_settings = {}
real_world_settings["method"] = refined_methods
real_world_settings["alpha"] = [1]
real_world_settings["clf"] = clf_keys
