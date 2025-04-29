"""
    Root Cause Localization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.public_function import aggregate_failure_representations, aggregate_instance_representations, ILD, SLD, save_pkl, load_pkl
from model.tasks.evaluation import eval_RCL
from sklearn.metrics.pairwise import cosine_similarity
from utils.public_function import save_pkl, load_pkl

def RCL(model, train_samples, test_samples, cases, split_ratio=0.6, top_k=3,
        before=300, after=300, verbose=False):
    service_dict_path = "Your-service_dict-path"
    service_dict_raw = load_pkl(service_dict_path)

    id_to_name = {v: k for k, v in service_dict_raw.items()}

    service_dict = {
        "name_to_id": service_dict_raw,
        "id_to_name": id_to_name
    }

    samples = train_samples + test_samples
    split_index = int(len(cases) * split_ratio)
    validation_cases = cases[split_index:] 

    ild_path = "ild.pkl"
    sld_path = "sld.pkl"
    instance_representations = aggregate_instance_representations(
        validation_cases, ILD(model, samples), before, after)
    save_pkl(instance_representations, ild_path)
    
    failure_representations, type_labels = aggregate_failure_representations(
        validation_cases, SLD(model, samples, top_k), None, before, after)
    save_pkl(failure_representations, sld_path)
    save_pkl(type_labels, "true_labels.pkl")
    result = []
    root_causes = []
    for idx, (case_data, case_rep) in enumerate(zip(validation_cases, instance_representations)):
        case_ts = case_data[0] 
        index_l1_sum = defaultdict(float)
        for index, d in case_rep:
            index_l1_sum[index] += np.sum(np.abs(d))
        sorted_rc = sorted(index_l1_sum.items(), key=lambda x: x[1], reverse=True)
        top_5_indices = [item[0] for item in sorted_rc[:5]]

        rc_names = [
            service_dict["id_to_name"].get(rc_id, f"Unknown_Service_{rc_id}")
            for rc_id in top_5_indices
        ]

        unknown_ids = [rc_id for rc_id in top_5_indices 
                      if rc_id not in service_dict["id_to_name"]]
        if unknown_ids and verbose:
            print(f"case {idx+1} found unknown ID: {unknown_ids}")

        result.append({
            "case_ts": case_ts,               
            "true_label": type_labels[idx],    
            "pred_rc_names": rc_names,         
        })
        root_causes.append(top_5_indices)

    accuracies = eval_RCL(root_causes, type_labels)
    
    return accuracies, result
