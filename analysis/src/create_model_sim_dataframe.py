import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src/")

from behavior_utils import *
from datautils import *

import pandas as pd
import warnings
from models import *
from datautils import *
from measures import *
from run_model import get_model


def get_trial_data_dict(subject_id, block_num, trial_num, trial):
    goal_progress_pre = trial["goal_progress_pre"]
    goal_progress_post = trial["goal_progress_post"]
    if "goal_selected" not in trial:
        trial["goal_selected"] = np.nan
    if "resource_selected" not in trial:
        trial["resource_selected"] = np.nan
    if "action_outcome" not in trial:
        trial["action_outcome"] = np.nan


    if trial['action_selected'] is not None:
        if int(block_num) < 6:
            session = 1
        else:
            session = 2
        trial["alien_selected"] = trial['action_selected'] + "_" + str(session)
    else:
        trial["alien_selected"] = np.nan

    trial_data = {
        "subject_id": subject_id,
        "block_num": int(block_num),
        "trial_num": int(trial_num), 
        #"block_type": get_block_type(int(trial["block_type"])),
        "block_type": trial["block_type"],
        "goal_selected": trial['goal_selected'],
        "subgoal_selected": trial['resource_selected'],
        "action_outcome": trial["action_outcome"],
        "alien_selected": trial["alien_selected"],
        "key_selected": trial["action_keypress"] if "action_keypress" in trial else None,

        "SH_progress": np.sum(np.array(list(goal_progress_pre["SH"].values()))[:,0]),
        "BR_progress": np.sum(np.array(list(goal_progress_pre["BR"].values()))[:,0]),
        "HO_progress": np.sum(np.array(list(goal_progress_pre["HO"].values()))[:,0]),


        "SH_progress_post": np.sum(np.array(list(goal_progress_post["SH"].values()))[:,0]),
        "BR_progress_post": np.sum(np.array(list(goal_progress_post["BR"].values()))[:,0]),
        "HO_progress_post": np.sum(np.array(list(goal_progress_post["HO"].values()))[:,0]),
        
        "G_status": trial["resources_pre"]["G"],
        "S_status": trial["resources_pre"]["S"],
        "M_status": trial["resources_pre"]["M"],
        "C_status": trial["resources_pre"]["C"],

        "G_progress_SH": goal_progress_pre["SH"]["G"][0] / goal_progress_pre["SH"]["G"][1],
        "M_progress_SH": goal_progress_pre["SH"]["M"][0] / goal_progress_pre["SH"]["M"][1],
        "C_progress_SH": np.nan,
        "S_progress_SH": np.nan,

        "G_progress_BR": goal_progress_pre["BR"]["G"][0] / goal_progress_pre["BR"]["G"][1],
        "S_progress_BR": goal_progress_pre["BR"]["S"][0] / goal_progress_pre["BR"]["S"][1],
        "C_progress_BR": goal_progress_pre["BR"]["C"][0] / goal_progress_pre["BR"]["C"][1],
        "M_progress_BR": np.nan,

        "G_progress_HO": np.nan,
        "S_progress_HO": goal_progress_pre["HO"]["S"][0] / goal_progress_pre["HO"]["S"][1],
        "C_progress_HO": goal_progress_pre["HO"]["C"][0] / goal_progress_pre["HO"]["C"][1],
        "M_progress_HO": goal_progress_pre["HO"]["M"][0] / goal_progress_pre["HO"]["M"][1],


        "G_progress_post_SH": goal_progress_post["SH"]["G"][0] / goal_progress_post["SH"]["G"][1],
        "M_progress_post_SH": goal_progress_post["SH"]["M"][0] / goal_progress_post["SH"]["M"][1],
        "C_progress_post_SH": np.nan,
        "S_progress_post_SH": np.nan,

        "G_progress_post_BR": goal_progress_post["BR"]["G"][0] / goal_progress_post["BR"]["G"][1],
        "S_progress_post_BR": goal_progress_post["BR"]["S"][0] / goal_progress_post["BR"]["S"][1],
        "C_progress_post_BR": goal_progress_post["BR"]["C"][0] / goal_progress_post["BR"]["C"][1],
        "M_progress_post_BR": np.nan,

        "G_progress_post_HO": np.nan,
        "S_progress_post_HO": goal_progress_post["HO"]["S"][0] / goal_progress_post["HO"]["S"][1],
        "C_progress_post_HO": goal_progress_post["HO"]["C"][0] / goal_progress_post["HO"]["C"][1],
        "M_progress_post_HO": goal_progress_post["HO"]["M"][0] / goal_progress_post["HO"]["M"][1],



        "G_prob": trial["G_prob"] if "G_prob" in trial else None,
        "S_prob": trial["S_prob"] if "S_prob" in trial else None,
        "M_prob": trial["M_prob"] if "M_prob" in trial else None,
        "C_prob": trial["C_prob"] if "C_prob" in trial else None,

        "goal_rt": trial["probe_rt"] if "probe_rt" in trial else None,
        "subgoal_rt": trial["action_rt"] if "action_rt" in trial else None,

        'goal_onset_jitter': trial["goal_onset_jitter"] if "goal_onset_jitter" in trial else None,
        'action_onset_jitter': trial["action_onset_jitter"] if "action_onset_jitter" in trial else None,

        'trial_start_time': None,
        'resource_status_onset_time': None,
        'goal_progress_onset_time': trial["goal_progress_onset_time"] if "goal_progress_onset_time" in trial else None,
        'goal_onset_time': trial["goal_onset_time"] if "goal_onset_time" in trial else None,
        'goal_probe_end_time': trial["goal_probe_end_time"] if "goal_probe_end_time" in trial else None,
        'goal_rt': trial["probe_rt"] if "probe_rt" in trial else None,
        'action_onset_time': trial["action_onset_time"] if "action_onset_time" in trial else None,
        'action_end_time': trial["action_end_time"] if "action_end_time" in trial else None,
        'action_rt': trial["action_rt"] if "action_rt" in trial else None,
        'outcome_onset_time': trial["outcome_onset_time"] if "outcome_onset_time" in trial else None,
        'resource_stack_onset_time': trial["resource_stack_onset_time"] if "resource_stack_onset_time" in trial else None,
        'trial_end_time': trial["trial_end_time"] if "trial_end_time" in trial else None,

    }
    return trial_data


class ModelSimDataFrame:
    def __init__(self, experiment=0, model_name="momentum_learn_alt_goal", data_type='fmri'):
        self.experiment = experiment
        self.model_name = model_name
        self.data_type = data_type


    def get_subject_dataframe(self, subject_id):
        """
            """
        if self.data_type == "online":
            results_path = MODEL_RESULTS_ONLINE
        else:
            results_path = MODEL_RESULTS

        data = get_subject_data_from_id(self.experiment, subject_id, data_type=self.data_type)
        file_name = results_path + self.model_name + "_" + str(self.experiment) + "/" + str(subject_id) + ".pkl"
        params = pickle.load(open(file_name, "rb"))

        model = get_model(self.model_name, params['params'])
        b_data = get_subject_data_from_id(self.experiment, subject_id,
                                        data_type=self.data_type)
        data = model.run_model(b_data)
        blocks = data['GoalSwitching']
        
        # Initialize a dataframe with column names
        df = pd.DataFrame()

        for block_num in list(blocks.keys()):
            block = blocks[block_num]
            trial_list = [key for key in block.keys() if 'trial_' in key]

            for trial_str in trial_list:
                trial = block[trial_str]
                trial_num = int(trial_str.split('_')[1])
                trial_data = get_trial_data_dict(subject_id, block_num, trial_num, trial)
                row_df = pd.DataFrame(trial_data, index=[0])
                warnings.simplefilter("ignore", FutureWarning)
                df = pd.concat([df, row_df], ignore_index=True)

        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])

        return df


            
def get_all_subjects_concatenated_dataframe(model_name, data_type='fmri'):
    """
    Returns a concatenated dataframe of all subjects for the specified experiment and data type.
    """
    all_subjects_df = pd.DataFrame()

    # Assuming you have a list of subject IDs
    if data_type == "online":
        subject_names = get_experiment_subjects(experiment=0, data_type=data_type)
        #active_subjects = range(len(subject_names))
        active_subjects = ACTIVE_SUBJECT_IDS_ONLINE
        print("Active online subjects:", active_subjects)
    elif data_type == "fmri":
        active_subjects = ACTIVE_SUBJECT_IDS 
    for subject_id in active_subjects:
        behavior_df = ModelSimDataFrame(experiment=0, data_type=data_type, model_name=model_name)
        subject_df = behavior_df.get_subject_dataframe(subject_id)
        all_subjects_df = pd.concat([all_subjects_df, subject_df], ignore_index=True)

    # save as csv
    all_subjects_df.to_csv(CACHE_DIR + "all_subjects_behavior_data_" + data_type + "_" + model_name + "_simulated.csv", index=False)
    return all_subjects_df
    








if __name__ == "__main__":
    # Example usage
    # behavior_df = BehaviorDataFrame(experiment=1, data_type='fmri')
    # subject_id = 10  # Replace with actual subject ID
    # df = behavior_df.get_subject_dataframe(subject_id)

    #model_name = "momentum_learn_alt_goal"
    #model_name = "momentum"
    #model_name = "prospective"
    #model_name = "td_persistence"
    #model_name = "retrospective"
    #model_name = "progress"
    model_name = "resources"
    model_name = "momentum_with_softmax"
    data_type = "online"
    get_all_subjects_concatenated_dataframe(model_name, data_type)

