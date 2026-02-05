import os
import json
import pickle
import platform
import numpy as np
print(platform.node())

if platform.node() == 'Snehas-MacBook-Air-5.local':
    FMRI_SUB_BEHAVIOR ='/Users/hypatia/Caltech/momentum_hierarchical_goal_pursuit/fmri/subject_data/'
    FMRI_BEHAVIOR = '/Users/hypatia/Caltech/momentum_hierarchical_goal_pursuit/analysis/'
    MODEL_RESULTS = '/Users/hypatia/Caltech/momentum_hierarchical_goal_pursuit/analysis/results/'
    MODEL_RESULTS_ONLINE = '/Users/hypatia/Caltech/momentum_hierarchical_goal_pursuit/analysis/results_online/'
    CACHE_DIR = "/Users/hypatia/Caltech/momentum_hierarchical_goal_pursuit/analysis/cache/"
    FIGURES = '/Users/hypatia/Caltech/momentum_hierarchical_goal_pursuit/analysis/figures/'
    FIGURES_TOPICS = '/Users/hypatia/Caltech/momentum_hierarchical_goal_pursuit/paper/Goal-switching-TOPICS/Goal--subgoal-switching---ToPICS/figures/'
    ONLINE_DATA = '/Users/hypatia/Caltech/momentum_hierarchical_goal_pursuit/data/'

else:
    FMRI_DERIVATIVES = '/resnick/groups/ODohertylab/saenugu/goal-switching/fw_subjects/data_bids/derivatives/fmriprep-23p1p3/'
    FMRI_SUB_BEHAVIOR = '/home/saenugu/Projects/momentum_hierarchical_goal_pursuit/fmri/subject_data/'
    FMRI_OUTPUTS = '/resnick/groups/ODohertylab/saenugu/goal-switching/fw_subjects/data_bids/derivatives/outputs/'
    FMRI_BEHAVIOR = '/home/saenugu/Projects/momentum_hierarchical_goal_pursuit/analysis/'
    MODEL_RESULTS = '/home/saenugu/Projects/momentum_hierarchical_goal_pursuit/analysis/results/'
    MODEL_RESULTS_NORM = '/home/saenugu/Projects/momentum_hierarchical_goal_pursuit/analysis/results_normalized/'
    MODEL_RESULTS_ONLINE = '/home/saenugu/Projects/momentum_hierarchical_goal_pursuit/analysis/results_online/'
    CACHE_DIR = "/home/saenugu/Projects/momentum_hierarchical_goal_pursuit/analysis/cache/"
    ONLINE_DATA = '/home/saenugu/Projects/momentum_hierarchical_goal_pursuit/data/'

#FMRI_SUBJECT_LIST = os.listdir(FMRI_SUB_BEHAVIOR)
ACTIVE_SUBJECT_IDS = list(set(np.arange(10, 43)) - set(np.array([22, 23, 27, 40])))
ACTIVE_SUBJECT_IDS_FMRI = list(set(np.arange(10, 43)) - set(np.array([27, 40])))
ACTIVE_SUBJECT_IDS_ONLINE = list(set(np.arange(30)) - set(np.array([])))

# def load_fmri_data_subject_input_json(subject_id, session_num):
#     sub_str = 'scan_0' + str(subject_id)

#     fmri_sub_dir = next((element for element in FMRI_SUBJECT_LIST if
#                          element.startswith(sub_str)), None)
#     fmri_sub_path = FMRI_SUB_BEHAVIOR + fmri_sub_dir + '/'

#     game_json_path = fmri_sub_path + "main_task_session_" + str(session_num) + ".json"
#     with open(game_json_path, 'r') as f:
#         game_json = json.load(f)
#     return game_json


# def load_fmri_data(subject_id, run_num, run_data=False):
#     sub_str = 'scan_0' + str(subject_id)

#     fmri_sub_dir = next((element for element in FMRI_SUBJECT_LIST if
#                          element.startswith(sub_str)), None)
#     fmri_sub_path = FMRI_SUB_BEHAVIOR + fmri_sub_dir + '/'

#     if not run_data:
#         data_pkl = fmri_sub_path + 'sub-' + str(subject_id) + '_run-' + str(
#             run_num) + '_data.pkl'
#         with open(data_pkl, 'rb') as f:
#             data = pickle.load(f)
#         return data
#     else:
#         run_data_pkl = fmri_sub_path + 'sub-' + str(subject_id) + '_run-' + str(
#             run_num) + '_run_data.pkl'
#         with open(run_data_pkl, 'rb') as f:
#             run_data = pickle.load(f)

#         return run_data


# def collate_subject_fmri_data_across_runs(subject_id, session_num):
#     data = {}
#     data['GoalSwitching'] = {}
#     data['run_start_times'] = []
#     data['run_end_times'] = []
#     data['run_block_nums']  = []
#     data['session_info'] = []

#     if session_num == 0:
#         runs = range(1, 4)
#     elif session_num == 1:
#         runs = range(4, 7)

#     for run_num in runs:
#         run_data = load_fmri_data(subject_id, run_num)
#         data['GoalSwitching'].update(run_data['GoalSwitching'])
#         data['run_start_times'].append(run_data['run_start_time'])
#         data['run_end_times'].append(run_data['run_end_time'])
#         data['run_block_nums'].append(run_data['GoalSwitching'].keys())

#     return data


def get_subject_data_from_id(experiment, subject_id, data_type='fmri'):
    if data_type == 'online':
        data_dir = ONLINE_DATA + "experiment_" + str(experiment) + "/"
        subject_names = get_experiment_subjects(experiment=experiment, data_type=data_type)
        sub_path = data_dir + subject_names[subject_id] + ".json"
        fp = open(sub_path)
        from process_behavioral_data import ProcessBehavioralData
        process_data = ProcessBehavioralData(json.load(fp))
        return process_data.process_data()
    # elif data_type == 'fmri':
    #     data = {}
    #     data_0 = collate_subject_fmri_data_across_runs(subject_id, session_num=0)
    #     data_1 = collate_subject_fmri_data_across_runs(subject_id, session_num=1)

    #     data['GoalSwitching'] = data_0['GoalSwitching']
    #     for key in data_1['GoalSwitching']:
    #         data['GoalSwitching'][str(int(key) + 6)] = data_1['GoalSwitching'][key]

        return data


def get_experiment_trial_details(experiment):
    details = {}

    if experiment == 0:
        details["num_episodes"] = 9
        details["num_trials"] = 33
        details["num_samples"] = 360

    return details

def check_subject_validity(experiment, subject_file):
    if subject_file.endswith('.json') == False:
        return None
    fp = open(subject_file)
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
    if data_type == 'online':
        data_path = ONLINE_DATA + "/experiment_" + str(experiment) + "/"
        subject_names = []
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

    #data = get_subject_data_from_id(experiment=0, sub_path=subject_names[0], data_type=data_type)
