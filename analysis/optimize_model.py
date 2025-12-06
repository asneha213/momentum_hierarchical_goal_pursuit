import sys
sys.path.append("..")
from src.models import *
from src.datautils import *
from src.measures import SubjectMeasure
from src.run_model import get_model

import numpy as np
import optuna
import pickle

from fit_behavior import get_optuna_params


class ModelOptimizer:
    def __init__(self, experiment, model_name, data_type='fmri'):
        self.experiment = experiment
        self.model_name = model_name
        self.data_type = data_type

    def get_mean_performance(self, params):
        seeds = np.random.randint(0, 1000, 30)
        model_results = []
        if self.data_type == 'fmri':
            active_subjects = ACTIVE_SUBJECT_IDS
        else:
            active_subjects = ACTIVE_SUBJECT_IDS_ONLINE
        for subject_id in active_subjects:
            model = get_model(self.model_name, params)
            data = get_subject_data_from_id(self.experiment, subject_id, data_type=self.data_type)
            model_res = model.run_model(data)
            measures = SubjectMeasure(subject_id=subject_id, experiment=self.experiment, model_res=model_res)
            sub_measure = measures.get_task_performance()
            model_results.append(sub_measure)
        return model_results

    def get_mean_measure(self, params, measure_name):
        model_results = []
        for subject_id in ACTIVE_SUBJECT_IDS:
            model = get_model(self.model_name, params)
            data = get_subject_data_from_id(self.experiment, subject_id, data_type=self.data_type)
            model_res = model.run_model(data)
            measures = SubjectMeasure(subject_id=subject_id, experiment=self.experiment, model_res=model_res)
            if measure_name == "optimal_goal":
                sub_measure = measures.get_optimal_goal_condition()
            elif measure_name == "goal_switches":
                sub_measure = measures.get_goal_switches_condition()
            elif measure_name == "goal_selects":
                sub_measure = measures.get_goal_selections_condition()
            model_results.append(sub_measure)
        return model_results

    def get_optuna_objective(self, trial):

        params = get_optuna_params(trial, self.model_name)
        model_measures = self.get_mean_performance(params)
        err = -1 * np.mean(model_measures)
        return err

    def fit_optuna(self):
        best_params = []
        best_vals = []
        for i in range(1):
            study = optuna.create_study()
            study.optimize(self.get_optuna_objective, n_trials=50)
            best_params.append(study.best_params)
            best_vals.append(study.best_value)

        best_params = np.array(best_params)
        best_vals = np.array(best_vals)
        best_params = best_params[np.argmin(best_vals)]
        best_vals = np.min(best_vals)
        print(best_vals, best_params)
        return best_params, best_vals

    def get_model_optimal_params(self):
    
        file_name = MODEL_RESULTS + "sims/" + self.model_name + "_" + str(self.experiment) + "_" + self.data_type + "_optimal_params.pkl"
        res = self.fit_optuna()
        with open(file_name, 'wb') as f:
            pickle.dump(res, f)
        return res


if __name__ == "__main__":

    experiment = 0
    model_name = "prospective"
    #model_name = "random"
    #model_name = "momentum"
    #model_name = "td_persistence"
    #model_name = "momentum_hierarchy"


    data_type = "online"



    # for model_name in ["momentum", "prospective", "td_persistence"]:
    #     res = ModelOptimizer(experiment, model_name).get_model_optimal_params()
    res = ModelOptimizer(experiment, model_name, data_type=data_type).get_model_optimal_params()
    print(res)

    # model_opt = ModelOptimizer(experiment, model_name)
    #
    # params = [0.5, 0.1, 0.1, 10, 10, 0.9]
    # model_measures = model_opt.get_mean_performance(params)
    # print(model_measures)
    # print(np.mean(model_measures))
    # #model_measures = model_opt.get_mean_measure(params, "goal_switches")
    # #model_measures = model_opt.get_mean_measure(params, "goal_selects")
    # print(model_measures)

