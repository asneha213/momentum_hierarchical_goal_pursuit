import pandas as pd
import numpy as np


def check_subgoal_completion(row):
    """
    Mark if subgoal is completed by the end of the trial.
    """
    goal = row['goal_selected']
    subgoal = row['subgoal_selected']
    
    col_pre = f"{subgoal}_progress_{goal}"
    col_post = f"{subgoal}_progress_post_{goal}"

    if col_pre in row and col_post in row:
        pre_val = row[col_pre]
        post_val = row[col_post]

        if not np.isclose(pre_val, 1.0) and np.isclose(post_val, 1.0):
            return 1
    return 0


def check_goal_completion(row):
    """
    Mark if goal is completed by the end of the trial.
    """
    goal = row['goal_selected']
    col_pre = f"{goal}_progress"
    col_post = f"{goal}_progress_post"

    if col_pre in row and col_post in row:
        pre_val = row[col_pre]
        post_val = row[col_post]

        if not np.isclose(pre_val, 1.0) and np.isclose(post_val, 1.0):
            return 1
    return 0

