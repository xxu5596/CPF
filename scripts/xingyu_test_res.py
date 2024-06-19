import os
import pickle
import time
import traceback

import numpy as np
import torch
import trimesh
from hocontact.hodatasets.cionline import CIOnline
from hocontact.hodatasets.ciquery import CIAdaptQueries, CIDumpedQueries
from hocontact.postprocess.geo_optimizer import GeOptimizer
from hocontact.utils.anatomyutils import AnatomyMetric
from hocontact.utils.collisionutils import penetration_loss_hand_in_obj, solid_intersection_volume
from hocontact.utils.disjointnessutils import region_disjointness_metric
from joblib import Parallel, delayed
from liegroups import SO3
from manopth.anchorutils import masking_load_driver
from manopth.quatutils import angle_axis_to_quaternion, quaternion_to_angle_axis
from termcolor import colored, cprint


def collapse_res_list(res_list_list):
    res = []
    for item in res_list_list:
        res.extend(item[0])
    return res


def merge_res_list(res_list):
    if len(res_list) < 1:
        return dict()

    keys_list = list(res_list[0].keys())
    #print(keys_list)

    # create init dict      all values=0
    res = {k: 0.0 for k in keys_list}
    #print(res)
    count = 0

    # iterate
    for item_id, item in enumerate(res_list):
        print(item_id, item)# 0,{'hand_dist_before': 0.1, 'hand_dist_after': 0.2, 'hand_joints_dist_before': 0.3, .....}
        for k in keys_list:
            print('k',k) # hand_dist_before
            print(item[k]) # 0.1
            if np.isnan(item[k]):
                cprint(f"encountered nan in {item_id} key {k}", "red")
                continue
            res[k] += item[k]
            print('res[k]',res[k])
        count += 1
    print('count',count)
    # avg
    for k in keys_list:
        res[k] /= count

    return res

res_list = [
    {
        "hand_dist_before": 0.1,
        "hand_dist_after": 0.2,
        "hand_joints_dist_before": 0.3,
        "hand_joints_dist_after": 0.4,
        "object_dist_before": 0.5,
        "object_dist_after": 0.6,
        "penetration_depth_gt": 0.7,
        "penetration_depth_before": 0.8,
        "penetration_depth_after": 0.9,
        "solid_intersection_volume_gt": 1.0,
        "solid_intersection_volume_before": 1.1,
        "solid_intersection_volume_after": 1.2,
        "disjointness_tip_only_gt": 1.3,
        "disjointness_tip_biased_gt": 1.4,
        "disjointness_tip_only_before": 1.5,
        "disjointness_tip_biased_before": 1.6,
        "disjointness_tip_only_after": 1.7,
        "disjointness_tip_biased_after": 1.8,
        "hand_kinetic_score_gt": 1.9,
        "hand_kinetic_score_before": 2.0,
        "hand_kinetic_score_after": 2.1,
    },
    {
        "hand_dist_before": 0.1,
        "hand_dist_after": 0.2,
        "hand_joints_dist_before": 0.3,
        "hand_joints_dist_after": 0.4,
        "object_dist_before": 0.5,
        "object_dist_after": 0.6,
        "penetration_depth_gt": 0.7,
        "penetration_depth_before": 0.8,
        "penetration_depth_after": 0.9,
        "solid_intersection_volume_gt": 1.0,
        "solid_intersection_volume_before": 1.1,
        "solid_intersection_volume_after": 1.2,
        "disjointness_tip_only_gt": 1.3,
        "disjointness_tip_biased_gt": 1.4,
        "disjointness_tip_only_before": 1.5,
        "disjointness_tip_biased_before": 1.6,
        "disjointness_tip_only_after": 1.7,
        "disjointness_tip_biased_after": 1.8,
        "hand_kinetic_score_gt": 1.9,
        "hand_kinetic_score_before": 2.0,
        "hand_kinetic_score_after": 2.1,
    },
    {
        "hand_dist_before": 0.1,
        "hand_dist_after": 0.2,
        "hand_joints_dist_before": 0.3,
        "hand_joints_dist_after": 0.4,
        "object_dist_before": 0.5,
        "object_dist_after": 0.6,
        "penetration_depth_gt": 0.7,
        "penetration_depth_before": 0.8,
        "penetration_depth_after": 0.9,
        "solid_intersection_volume_gt": 1.0,
        "solid_intersection_volume_before": 1.1,
        "solid_intersection_volume_after": 1.2,
        "disjointness_tip_only_gt": 1.3,
        "disjointness_tip_biased_gt": 1.4,
        "disjointness_tip_only_before": 1.5,
        "disjointness_tip_biased_before": 1.6,
        "disjointness_tip_only_after": 1.7,
        "disjointness_tip_biased_after": 1.8,
        "hand_kinetic_score_gt": 1.9,
        "hand_kinetic_score_before": 2.0,
        "hand_kinetic_score_after": 2.1,
    }
    # Add more dictionaries here if needed
]

# Merging the list of dictionaries
merged_res = merge_res_list(res_list)

# Printing the merged results
print(merged_res)




'''
res_list = [
            res1,
            res2,
            res3,
            ...
]


res1={'key':value}
res1=
    {
        "hand_dist_before": 0.1,
        "hand_dist_after": 0.2,
        "hand_joints_dist_before": 0.3,
        "hand_joints_dist_after": 0.4,
        "object_dist_before": 0.5,
        "object_dist_after": 0.6,
        "penetration_depth_gt": 0.7,
        "penetration_depth_before": 0.8,
        "penetration_depth_after": 0.9,
        "solid_intersection_volume_gt": 1.0,
        "solid_intersection_volume_before": 1.1,
        "solid_intersection_volume_after": 1.2,
        "disjointness_tip_only_gt": 1.3,
        "disjointness_tip_biased_gt": 1.4,
        "disjointness_tip_only_before": 1.5,
        "disjointness_tip_biased_before": 1.6,
        "disjointness_tip_only_after": 1.7,
        "disjointness_tip_biased_after": 1.8,
        "hand_kinetic_score_gt": 1.9,
        "hand_kinetic_score_before": 2.0,
        "hand_kinetic_score_after": 2.1,
    }
....

here the aim is to average all the values of each key. item[k]=value
'''
def summarize(res_dict):
    for k, v in res_dict.items():
        print("mean " + str(k), v)

summarize(merged_res)
'''
output:
mean hand_dist_before 0.10000000000000002
mean hand_dist_after 0.20000000000000004
mean hand_joints_dist_before 0.3
mean hand_joints_dist_after 0.4000000000000001
mean object_dist_before 0.5
mean object_dist_after 0.6
mean penetration_depth_gt 0.6999999999999998
mean penetration_depth_before 0.8000000000000002
mean penetration_depth_after 0.9
mean solid_intersection_volume_gt 1.0
mean solid_intersection_volume_before 1.1
mean solid_intersection_volume_after 1.2
mean disjointness_tip_only_gt 1.3
mean disjointness_tip_biased_gt 1.3999999999999997
mean disjointness_tip_only_before 1.5
mean disjointness_tip_biased_before 1.6000000000000003
mean disjointness_tip_only_after 1.7
mean disjointness_tip_biased_after 1.8
mean hand_kinetic_score_gt 1.8999999999999997
mean hand_kinetic_score_before 2.0
mean hand_kinetic_score_after 2.1
'''