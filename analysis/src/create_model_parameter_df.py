import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src/")

from models import *
from run_model import get_model
import pandas as pd
from datautils import *
from create_behavior_dataframe import get_trial_data_dict
from missing_data_handler import *
import warnings



class ModelParameterDF:
    def __init__(self, subject_id, model_name, model_params=None, run_num=None, data_type='fmri', optimal=False):
        experiment = 0
        self.subject_id = subject_id
        self.run_num = run_num
        self.data_type = data_type
        if run_num is not None:
            self.subject_data = load_fmri_data(subject_id, run_num)
        else:
            self.subject_data = get_subject_data_from_id(experiment, subject_id, data_type=data_type)
        if data_type == 'fmri':
            self.subject_data = fill_in_goal_completion_trials(subject_id, self.subject_data, run_num=run_num)
        self.model_name = model_name
        if model_params is None:
            if optimal:
                self.model_params = self.load_optimal_params(experiment)
            else:
                self.model_params = self.load_params(experiment, subject_id)
        else:
            self.model_params = model_params
        print(f"Model parameters for subject {subject_id} in experiment {experiment}: {self.model_params}")


    def load_optimal_params(self, experiment):
        file_name = MODEL_RESULTS + "sims/" + self.model_name + "_" + str(experiment) + "_" + self.data_type + "_optimal_params.pkl"
        with open(file_name, 'rb') as f:
            model_params = pickle.load(f)
            optimal_params = model_params[0]
        return optimal_params

        
    def load_params(self, experiment, subject_id):
        if self.data_type == 'fmri':
            subject_path = MODEL_RESULTS + self.model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl"
        else:
            subject_path = MODEL_RESULTS_ONLINE + self.model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl"
        with open(subject_path, "rb") as f:
            model_fits = pickle.load(f)
        return model_fits['params']

    def create_df(self, simulate=False):
        model = get_model(self.model_name, self.model_params)
        model.reset_slots()
        model.reset_card_probs()
        blocks_data = self.subject_data['GoalSwitching']
        df = pd.DataFrame()
        block_keys = list(blocks_data.keys())
        #sort block keys to ensure order
        block_keys.sort(key=lambda x: int(x))
        print(block_keys)
        prev_outcome_onset_time = None
        for block_num in block_keys:
            block = blocks_data[str(block_num)]
            model.reset_card_probs()

            trial_list = [int(i.split("_")[1]) for i in block.keys() if "trial_" in i]
            trial_list = sorted(trial_list)
            max_trial_num = max(trial_list)

            if block_num == '6':
                self.prev_goal = None
                self.prev_subgoal = None
                self.prev_outcome_onset_time = None
            for trial_num in range(max_trial_num + 1):
                model.trial_num = trial_num
                trial = block['trial_' + str(trial_num)]
                trial_dict = get_trial_data_dict(self.subject_id, block_num, trial_num, trial)
                if prev_outcome_onset_time is not None:
                    if trial_dict['goal_onset_time'] - prev_outcome_onset_time < 8:
                        trial_dict['prev_outcome_onset_time'] = prev_outcome_onset_time
                    else:
                        trial_dict['prev_outcome_onset_time'] = None
                else:
                    trial_dict['prev_outcome_onset_time'] = None
                if simulate:
                    trial_m, _, _, trial_dict = model.run_sim_subject_action_trial(trial, trial_dict=trial_dict)
                else:
                    trial_m, _, _,_, trial_dict = model.run_subject_action_trial(trial, trial_dict=trial_dict)
                row_df = pd.DataFrame(trial_dict, index=[0])
                if row_df.empty:
                    print(f"Empty row for subject {self.subject_id}, block {block_num}, trial {trial_num}")
                warnings.simplefilter("ignore", FutureWarning)
                df = pd.concat([df, row_df], ignore_index=True)
                if 'action_outcome' in trial_dict:
                    if trial_dict['action_outcome'] is not None:
                        prev_outcome_onset_time = trial_dict['outcome_onset_time']
                    else:
                        prev_outcome_onset_time = None
                else:
                    prev_outcome_onset_time = None

        df = df.sort_values(by=['subject_id', 'block_num', 'trial_num'])
        df['run_num'] = (np.floor(df['block_num'] / 2) + 1).astype(int)
        return df



def get_all_subjects_concatenated_model_dataframe(data_type='online', model_name="momentum_learn_alt_goal", optimal=False, simulate=False, read_from_csv=False):
    """
    Returns a concatenated dataframe of all subjects for the specified experiment and data type.
    """
    if read_from_csv:
        if model_name == "momentum_learn_alt_goal" and not optimal:
            all_subjects_df = pd.read_csv(f"all_subjects_model_parameters_{data_type}.csv")
        else:
            all_subjects_df = pd.read_csv(f"all_subjects_model_parameters_{model_name}_{data_type}_optimal_{optimal}.csv")
        return all_subjects_df
    all_subjects_df = pd.DataFrame()

    # Assuming you have a list of subject IDs
    if data_type == "online":
        subject_names = get_experiment_subjects(experiment=0, data_type=data_type)
        #active_subjects = range(len(subject_names))
        active_subjects = ACTIVE_SUBJECT_IDS_ONLINE
        print("Active online subjects:", active_subjects)
    elif data_type == "fmri":
        active_subjects = ACTIVE_SUBJECT_IDS_FMRI
    for subject_id in active_subjects:
        model_df = ModelParameterDF(subject_id, model_name=model_name, data_type=data_type, optimal=optimal)
        subject_df = model_df.create_df(simulate=simulate)
        all_subjects_df = pd.concat([all_subjects_df, subject_df], ignore_index=True)

    if simulate:
        #model_name += "_simulated"
        all_subjects_df.to_csv(f"all_subjects_model_parameters_{model_name}_{data_type}_simulated.csv", index=False)
    else:
        if model_name == "momentum_learn_alt_goal" and not optimal:
            all_subjects_df.to_csv(f"all_subjects_model_parameters_{data_type}.csv", index=False)
        else:
            all_subjects_df.to_csv(f"all_subjects_model_parameters_{model_name}_{data_type}_optimal_{optimal}.csv", index=False)

    return all_subjects_df



if __name__ == "__main__":
    # Example usage
    subject_id = 19
    model_name = "momentum_learn_alt_goal"
    #model_name = "prospective"

    #optimal = True
    optimal = False
    simulate = False
    
    
    #model_df = ModelParameterDF(subject_id, model_name)
    #model_df.create_df()

    all_subjects_df = get_all_subjects_concatenated_model_dataframe(data_type='online', model_name=model_name,\
                                                                     optimal=optimal, simulate=simulate)
    # Further processing can be done with the created DataFrame