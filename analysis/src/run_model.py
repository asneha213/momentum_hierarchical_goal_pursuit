"""Model-name dispatch and simulation helpers shared by fitting and analysis."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src/")


from models.prospective import Prospective
from models.prospective_exp import ProspectiveExp
from models.momentum import Momentum
from models.td_persistence import TDPersistence
from models.retrospective import Retrospective
from models.resources import Resources
from models.resources_depth import ResourcesDepth
from models.progress import Progress
from models.momentum_with_pers_subgoal import MomentumWithPersSubgoal
from models.momentum_with_pros_subgoal import MomentumWithProsSubgoal
from models.momentum_with_softmax import MomentumWithSoftmax


from measures import *

import pickle
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


from datautils import *


def get_model(model_name, params):
    if "_normalized" in model_name:
        model_name = model_name.replace("_normalized", "")
    if model_name == "prospective":
        return Prospective(params)
    elif model_name == "prospective_exp":
        return ProspectiveExp(params)
    elif model_name == "retrospective":
        return Retrospective(params)
    elif model_name == "resources":
        return Resources(params)
    elif model_name == "resources_depth":
        return ResourcesDepth(params)
    elif model_name == "progress":
        return Progress(params)
    elif model_name == "momentum":
        return Momentum(params, learn_alt_goal=False)
    elif model_name == "momentum_learn_alt_goal":
        return Momentum(params, learn_alt_goal=True)
    elif model_name == "momentum_with_pers_subgoal":
        return MomentumWithPersSubgoal(params, learn_alt_goal=True)
    elif model_name == "momentum_with_pros_subgoal":
        return MomentumWithProsSubgoal(params, learn_alt_goal=True)
    elif model_name == "momentum_with_softmax":
        return MomentumWithSoftmax(params, learn_alt_goal=True)
    elif model_name == "td_persistence":
        return TDPersistence(params)


def get_aggregate_model_measures(experiment, model_name, measure_name, cache=False):
    """
    Get the aggregate model measures for a given model and measure
            :param experiment: The experiment number
            :param model_name: The model name
            :param measure_name: The measure name
            :param cache: Whether to cache the results
            :return: The aggregate model measures
    """
    cache_file_name = CACHE_DIR + model_name + "_" + str(experiment) + "_" + measure_name + ".npy"
    if cache:
        measures = np.load(cache_file_name)
        return measures
    model_measures = []
    for subject_id in ACTIVE_SUBJECT_IDS:
        file_name = MODEL_RESULTS + model_name + "_" + str(experiment) + "/" + str(subject_id) + ".pkl"
        params = pickle.load(open(file_name, "rb"))

        model = get_model(model_name, params['params'])
        data = get_subject_data_from_id(experiment, subject_id,
                                        data_type='fmri')
        model_res = model.run_model(data)
        subject_measures = SubjectMeasure(subject_id=subject_id, experiment=experiment, model_res=model_res)
        model_measures.append(get_behavioral_measure(subject_measures, measure_name))
    model_measures = np.array(model_measures)
    np.save(cache_file_name, model_measures)
    return model_measures


