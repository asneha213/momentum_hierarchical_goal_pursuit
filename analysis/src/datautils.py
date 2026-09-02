"""Shared repository paths, participant selection, and raw-record loading."""
import os
import json
import pickle
from pathlib import Path
import numpy as np

# Callers concatenate filenames, so these directory strings retain a trailing slash.
_ROOT = Path(__file__).resolve().parents[2]
FMRI_SUB_BEHAVIOR = str(_ROOT / 'fmri' / 'subject_data') + '/'
FMRI_BEHAVIOR = str(_ROOT / 'analysis') + '/'
MODEL_RESULTS = str(_ROOT / 'analysis' / 'results') + '/'
MODEL_RESULTS_NORM = str(_ROOT / 'analysis' / 'results_normalized') + '/'
MODEL_RESULTS_ONLINE = str(_ROOT / 'analysis' / 'results_online') + '/'
CACHE_DIR = str(_ROOT / 'analysis' / 'cache') + '/'
FIGURES = str(_ROOT / 'analysis' / 'figures') + '/'
FIGURES_TOPICS = FIGURES  # The reproduction runner overrides this for each run.
ONLINE_DATA = str(_ROOT / 'data') + '/'

# Retain the historical participant exclusions and integer IDs used by saved fits.
ACTIVE_SUBJECT_IDS = list(set(np.arange(10, 43)) - set(np.array([22, 23, 27, 40])))
ACTIVE_SUBJECT_IDS_FMRI = list(set(np.arange(10, 43)) - set(np.array([27, 40])))
ACTIVE_SUBJECT_IDS_ONLINE = list(set(np.arange(30)) - set(np.array([])))

def get_subject_data_from_id(experiment, subject_id, data_type='fmri'):
    """Load one online record and reconstruct its trial state for analysis."""
    if data_type == 'online':
        data_dir = ONLINE_DATA + "experiment_" + str(experiment) + "/"
        subject_names = get_experiment_subjects(experiment=experiment, data_type=data_type)
        sub_path = data_dir + subject_names[subject_id] + ".json"
        from process_behavioral_data import ProcessBehavioralData
        with open(sub_path) as fp:
            process_data = ProcessBehavioralData(json.load(fp))
        return process_data.process_data()


def get_experiment_trial_details(experiment):
    details = {}

    if experiment == 0:
        details["num_episodes"] = 9
        details["num_trials"] = 33
        # Historical full-block BIC sample-size convention (not recomputed here).
        details["num_samples"] = 360

    return details

def check_subject_validity(experiment, subject_file):
    if subject_file.endswith('.json') == False:
        return None
    with open(subject_file) as fp:
        data = json.load(fp)
    if "GoalSwitching" not in data:
        return None
    details = get_experiment_trial_details(experiment)


    for i in range(details["num_episodes"]):
        if str(i) not in data['GoalSwitching']:
            return None
        for j in range(details["num_trials"]):
            if 'trial_'+ str(j) not in data['GoalSwitching'][str(i)]:
                return None
    return data


def get_experiment_subjects(experiment, data_type='fmri'):
    """Return valid record names in the historical filesystem enumeration order."""
    if data_type == 'online':
        data_path = ONLINE_DATA + "/experiment_" + str(experiment) + "/"
        subject_names = []
        # Do not sort here: numbered fits refer to the original listing order.
        # Moving datasets requires checking the participant-to-file mapping.
        for subject_file in os.listdir(data_path):
            valid = check_subject_validity(experiment, data_path + subject_file)
            if valid:
                subject_names.append(subject_file.strip('.json'))

        return subject_names
    elif data_type == 'fmri':
        subject_names = [element for element in FMRI_SUBJECT_LIST if element.startswith('scan_')]
        return subject_names


def load_design_file(subject_id, run):

    sub_str = 'scan_0' + str(subject_id)

    fmri_sub_dir = next((element for element in FMRI_SUBJECT_LIST if
                     element.startswith(sub_str)), None)
    fmri_sub_path = FMRI_SUB_BEHAVIOR + fmri_sub_dir + '/'

    if run < 4:
        design_file = fmri_sub_path + "main_task_session_1.json"
    else:
        design_file = fmri_sub_path + "main_task_session_2.json"

    with open(design_file, 'r') as f:
        design = json.load(f)
    return design



if __name__ == "__main__":
    data_type = "online"
    subject_names = get_experiment_subjects(experiment=0, data_type=data_type)
    print(len(subject_names))
    print(subject_names)
