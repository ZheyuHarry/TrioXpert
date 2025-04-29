"""
    Evaluate the downstream task of AD, FT, RCL
"""

import numpy as np

###########################################
##              EVAL——AD                 ##
###########################################

def eval_AD(pre_interval, ad_cases_label, impact_window=5*60, verbose=False):
    # match
    pre_dict = {key: set() for key in pre_interval}
    ad_dict = {key: set() for key in ad_cases_label}
    print(f'length of prediction = {len(pre_dict)}')
    print(f'length of truth = {len(ad_dict)}')
    for s, e in pre_interval:
        for case_ts in ad_cases_label:
            case_s, case_e = case_ts-impact_window, case_ts+impact_window
            if not (case_s > e or case_e < s): # Indicate that the predicted interval (s , e) is legal
                pre_dict[(s, e)].add(case_ts)
                ad_dict[case_ts].add((s, e))
    # calculate 
    TP = len([key for key,value in ad_dict.items() if len(value) > 0]) # Anomolous and predicted as anomaly
    FP = len([key for key,value in pre_dict.items() if len(value) == 0]) # Not anomolous but predicted as anomaly
    FN = len([key for key,value in ad_dict.items() if len(value) == 0]) # Anomolous but predicted as normal
    precision = np.round(TP / (TP + FP), 4)
    recall = np.round(TP / (TP + FN), 4)
    f1 = np.round(2 * precision * recall / (precision + recall), 4)
    density = np.round(np.mean([len(value) for key,value in pre_dict.items() if len(value) > 0]), 2)
    if verbose:
        print(f'precision: {precision}, recall: {recall}, f1: {f1}, density: {density}')
    return precision, recall, f1

###########################################
##              EVAL——RCL                ##
###########################################

def eval_RCL(root_causes, labels):
    k_values = [1, 3, 5]
    accuracies = {k: 0 for k in k_values}
    
    for pred, true_labels in zip(root_causes, labels):
        print(f"Pred are {pred}     &     true_labels are {true_labels}")
        if pred is None:  
            continue
        
        min_pos = float('inf')
        for label in true_labels:
            if label in pred:
                pos = pred.index(label)
                if pos < min_pos:
                    min_pos = pos
        
        if min_pos != float('inf'):
            for k in k_values:
                if k >= (min_pos + 1):  
                    accuracies[k] += 1
    n_samples = len(root_causes)
    for k in k_values:
        accuracies[k] /= n_samples
    
    return accuracies

###########################################
##              EVAL——FT                 ##
###########################################
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter
def eval_FT(labels, pre):
    precision = np.round(precision_score(labels, pre), 4)
    recall = np.round(recall_score(labels, pre), 4)
    f1 = np.round(f1_score(labels, pre), 4)
    return precision, recall, f1


def print_channel_detials(node, channel_dict):
    if node is None:
        return
    
    indent = '     ' * node.depth
    
    if isinstance(channel_dict, dict) and "id_to_name" in channel_dict:
        try:
            split_dim_int = int(node.split_dim)
            channel_name = channel_dict["id_to_name"].get(split_dim_int, f"Unknown Channel ({split_dim_int})")
        except ValueError:
            channel_name = f"Invalid Split Dimension: {node.split_dim}"
            print(f"Error: Unable to convert {node.split_dim} to integer.")
    else:
        print("Warning: channel_dict is not a valid dictionary or missing 'id_to_name' key.")
        channel_name = f"Unknown Channel ({node.split_dim})"
    
    if node.left is None and node.right is None:
        print(indent + f' | Split Dimension: {node.split_dim}, Split Criteria: {node.criteria}, Split Value: {node.split_value}, Num Vectors: {len(node.failure_infos)}, In distance: {node.in_distance}')
        print(indent + ' * ' + f'[{node.label_id}] ' + str(Counter([failure_info.label for failure_info in node.failure_infos])))
    else:
        print(indent + f'Split Dimension: {node.split_dim}, Split Criteria: {node.criteria}, Split Value: {node.split_value}, Num Vectors: {len(node.failure_infos)}, In distance: {node.in_distance}, {channel_name}')
    
    print_channel_detials(node.left, channel_dict)
    print_channel_detials(node.right, channel_dict)
