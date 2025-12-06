def get_action_relevance_in_goal(goal, action, resources):
    if goal == "SH":
        if action == "S" or action == "C":
            return 0
        elif action == "M":
            if resources["M"] < 3:
                return 1
            else:
                return 0
        elif action == "G":
            if resources["G"] < 3:
                return 1
            else:
                return 0

    elif goal == "HO":
        if action == "G":
            return 0
        elif action == "S":
            if resources["S"] < 1:
                return 1
            else:
                return 0
        elif action == "M":
            if resources["M"] < 2:
                return 1
            else:
                return 0
        elif action == "C":
            if resources["C"] < 3:
                return 1
            else:
                return 0

    elif goal == "BR":
        if action == "M":
            return 0
        elif action == "S":
            if resources["S"] < 3:
                return 1
            else:
                return 0
        elif action == "G":
            if resources["G"] < 2:
                return 1
            else:
                return 0
        elif action == "C":
            if resources["C"] < 1:
                return 1
            else:
                return 0


def get_action_relevance_from_goal_progress(goal, action, goal_progress):
    if goal == "SH":
        if action == "S" or action == "C":
            return 0
        elif action == "M":
            if goal_progress[goal]["M"][0] < 3:
                return 1
            else:
                return 0
        elif action == "G":
            if goal_progress[goal]["G"][0] < 3:
                return 1
            else:
                return 0

    elif goal == "HO":
        if action == "G":
            return 0
        elif action == "S":
            if goal_progress[goal]["S"][0] < 1:
                return 1
            else:
                return 0
        elif action == "M":
            if goal_progress[goal]["M"][0] < 2:
                return 1
            else:
                return 0
        elif action == "C":
            if goal_progress[goal]["C"][0] < 3:
                return 1
            else:
                return 0

    elif goal == "BR":
        if action == "M":
            return 0
        elif action == "S":
            if goal_progress[goal]["S"][0] < 3:
                return 1
            else:
                return 0
        elif action == "G":
            if goal_progress[goal]["G"][0] < 2:
                return 1
            else:
                return 0
        elif action == "C":
            if goal_progress[goal]["C"][0] < 1:
                return 1
            else:
                return 0


def get_num_relevant_actions_goal(goal, resources):
    num_relevant = 0
    for action in ["G", "C", "M", "S"]:
        if get_action_relevance_in_goal(goal, action, resources) == 1:
            num_relevant += 1
    return num_relevant

def get_num_relevant_actions_goal_progress(goal, goal_progress):
    num_relevant = 0
    for action in ["G", "C", "M", "S"]:
        if get_action_relevance_from_goal_progress(goal, action, goal_progress) == 1:
            num_relevant += 1
    return num_relevant


def get_relevant_goals_for_subgoals():
    subgoal_rel_goal = {
        "G": ["SH", "BR"],
        "C": ["HO", "BR"],
        "M": ["SH", "HO"],
        "S": ["BR", "HO"]
    }
    return subgoal_rel_goal



def get_relevant_alternative_subgoals_for_subgoals():
    """
    Get subgoals of other goals that are goes together with the subgoal of a given goal
    """
    subgoal_rel_subgoal = {
        ("SH", "G"): ["S", "C"],
        ("BR", "G"): ["M"],
        ("BR", "C"): ["M", "S", "C"],
        ("HO", "C"): ["S", "G"],
        ("SH", "M"): ["S", "C"],
        ("HO", "M"): ["G"],
        ("BR", "S"): ["C", "M"],
        ("HO", "S"): ["S", "C", "G"]
    }

    return subgoal_rel_subgoal

def get_relevant_congruent_subgoals_for_subgoal():
    """
    Get the other subgoals of the same goal for a given subgoal

    """
    subgoal_rel_subgoal = {
        ("SH", "G"): ["M"],
        ("BR", "G"): ["S", "C"],
        ("BR", "C"): ["S", "G"],
        ("HO", "C"): ["M", "S"],
        ("SH", "M"): ["G"],
        ("HO", "M"): ["S", "C"],
        ("BR", "S"): ["G", "C"],
        ("HO", "S"): ["M", "C"]
    }

    return subgoal_rel_subgoal



def get_subgoal_progress_given_goal_progress(goal_progress):
    subgoal_progress = {'SH': {}, 'HO': {}, 'BR': {}}

    for goal in subgoal_progress.keys():
        for action in ['G', 'C', 'M', 'S']:
            if action in goal_progress[goal].keys():
                subgoal_progress[goal][action] = goal_progress[goal][action][0]/goal_progress[goal][action][1]
            else:
                subgoal_progress[goal][action] = -1
    return subgoal_progress


def get_max_subgoal_progress(goal_progress):
    subgoal_progress = {'M': {'SH': 0, 'HO': 0},
                        'G': {'SH': 0, 'BR': 0},
                        'C': {'HO': 0, 'BR': 0},
                        'S': {'HO': 0, 'BR': 0}}

    for subgoal in subgoal_progress.keys():
        for goal in subgoal_progress[subgoal].keys():
            subgoal_progress[subgoal][goal] = goal_progress[goal][subgoal][0] / goal_progress[goal][subgoal][1]

    # get max subgoal progress for each subgoal
    max_subgoal_progress = {}
    for subgoal in subgoal_progress.keys():
        max_subgoal_progress[subgoal] = max(subgoal_progress[subgoal].values())


    return max_subgoal_progress
