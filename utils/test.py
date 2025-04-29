"""
    We define the test function here
"""
import torch
from utils.public_function import load_pkl
from model.tasks.anomaly_detection import AD
from model.tasks.root_cause_localization import RCL
from model.tasks.failure_triage import FT
import pandas as pd
def test(model, model_path, train_path, test_path,gt_path, mode=['AD', 'FT', 'RCL'], device='cpu', split_ratio=0.3, before=300, after=300):
    train_samples = load_pkl(train_path)
    test_samples = load_pkl(test_path)
    cases = load_pkl(gt_path)

    # Load the trained model
    model.load_state_dict(torch.load(model_path))
    
    if 'AD' in mode:
        result = []
        pre_interval,case_ts, precision, recall , f1 = AD(model, train_samples, test_samples, cases=cases, split_ratio=0.7, delay=600, impact_window=300)
        result.append(pre_interval)
        result.append(case_ts)
        print(f'precision = {precision}')
        print(f'recall = {recall}')
        print(f'f1 = {f1}')
    if 'RCL' in mode:
        accuracies,result= RCL(model, train_samples, test_samples, cases, split_ratio=split_ratio, before=before, after=after)
        print(accuracies)
    if 'FT' in mode:
        type_hash=load_pkl("Your-type_hash-path")
        type_dict=load_pkl("Your-type_dict-path")

        samples = train_samples + test_samples
        pre_types, precision, recall, f1 = FT(
            model, samples, cases, type_hash=type_hash ,type_dict= type_dict, split_ratio=0.7, t_value=3, before=59, after=300, max_clusters=50, channel_dict=None, verbose=False
        )

        target_cases = cases[int(len(cases)*split_ratio):]
        assert len(target_cases) == len(pre_types)

        print("\nFT Mode Results:")
        print(f"- Precision: {precision}")
        print(f"- Recall: {recall}")
        print(f"- F1 Score: {f1}")
        print(f"- Predicted Types Sample: {pre_types}")
        result = {'pre_types': pre_types,'target_cases': target_cases}
    return result