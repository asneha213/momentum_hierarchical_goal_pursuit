import numpy as np
import pandas as pd
from itertools import combinations


A_G = {
    "SH" : {"G": 3, "M": 3, "S": 0, "C": 0},
    "HO" : {"M": 2, "C": 3, "S": 1, "G": 0},
    "BR" : {"G": 2, "S": 3, "C": 1, "M": 0},
}


def get_goal_probs_from_resources(aprobs):
    gprobs = {}
    for g in ["SH", "HO", "BR"]:
        gprobs[g] = 0
        for r in ["G", "M", "S", "C"]:
            gprobs[g] += aprobs[r] * A_G[g][r]
        gprobs[g] = gprobs[g] / 6

    print(gprobs)
    return gprobs


if __name__ == "__main__":

    mode = "HO"
    condition = "extreme"
    #condition = "moderate"

    # H Higher Others Lower
    # SH, HO
    if mode == "SH":
        if condition == "extreme":
            aprobs = {"M": 0.8, "C": 0.25, "S": 0.25, "G": 0.8}

        else:
            aprobs = {"M": 0.6, "C": 0.5, "S": 0.4, "G": 0.6}

    # P Higher Others Lower
    # BR, SH
    if mode == "BR":
        if condition == "extreme":
            aprobs = {"M": 0.2, "C": 0.5, "S": 0.8, "G": 0.8}
        else:
            aprobs = {"M": 0.4, "C": 0.5, "S": 0.6, "G": 0.6}

    # B Higher Others Lower
    # BR, HO
    if mode == "HO":
        if condition == "extreme":
            aprobs = {"M": 0.8, "C": 0.8, "S": 0.5, "G": 0.2}

        else:
            aprobs = {"M": 0.6, "C": 0.6, "S": 0.5, "G": 0.4}

    goal_values = get_goal_probs_from_resources(aprobs)
    goal_values = list(goal_values.values())
    goal_values.sort()
    print(goal_values[-1] - goal_values[1])

