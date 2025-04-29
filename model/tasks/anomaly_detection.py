"""
    Anomaly Detection
"""

from utils.public_function import SLD, get_threshold
from model.tasks.evaluation import eval_AD
import numpy as np

def get_pre_interval(eval_slds_sum, delay=600):
    pre_ts_df = eval_slds_sum[eval_slds_sum['outlier']==1] # Filter those are outliers
    pre_ts_df['diff'] = [0] + np.diff(pre_ts_df['timestamp']).tolist() # Calculate the difference between the adjcent timestamps
    pre_interval = []
    start_ts, end_ts = None, None
    for _, span in pre_ts_df.iterrows():
        if start_ts is None:
            start_ts = int(span['timestamp'])
        if span['diff'] >= delay:  
            pre_interval.append((start_ts, end_ts))
            start_ts = int(span['timestamp'])
        end_ts = int(span['timestamp'])
    pre_interval.append((start_ts, end_ts)) # 
    # filter
    pre_interval = [(item[0], item[1]) for item in pre_interval if item[0]!=item[1]]
    return pre_interval

def AD(model, train_samples, test_samples, cases, split_ratio=0.6, delay=600, impact_window = 300, verbose=True):
    total_samples = train_samples + test_samples
    total_samples.sort(key=lambda x: x[0])
    ts_list = sorted([item[0] for item in total_samples])
    # Get the split timestamp
    split_ts = ts_list[int(len(ts_list)*split_ratio)]    

    case_ts = [case[0] for case in cases if case[0] > split_ts] 
    init_samples = [item for item in train_samples if item[0] <= split_ts]
    eval_samples = [item for item in total_samples if item[0] > split_ts]
    init_slds = SLD(model, init_samples, top_k=3)
    eval_slds = SLD(model, eval_samples, top_k=3)
    print(f'init slds.shape = {init_slds.shape}')
    print(f'eval slds.shape = {eval_slds.shape}')
    
    init_slds_sum = init_slds.set_index('timestamp').sum(axis=1).reset_index() # Set the "timestamp" column as index & Calculate the samples' each SLD L1-Norm
    eval_slds_sum = eval_slds.set_index('timestamp').sum(axis=1).reset_index()
    init_slds_sum_list = init_slds_sum.iloc[:, 1].tolist()
    
    threshold = get_threshold(init_slds_sum_list, q=0.21) 
    print(f'threshold = {threshold}')
    eval_slds_sum.to_csv("results/eval_sld_sums.csv")
    if threshold > 0:
        eval_slds_sum['outlier'] = eval_slds_sum[0].apply(lambda x: 1 if x > threshold else 0)
        if verbose:
            print(f'threshold is {threshold}.')
            # Calculate the outlier ratio according to the "outlier" column
            print(f"outlier ratio is {np.round(sum(eval_slds_sum['outlier']==1)/len(eval_slds_sum), 4)}.")
        pre_interval = get_pre_interval(eval_slds_sum, delay)
        precision, recall, f1 = eval_AD(pre_interval, case_ts, impact_window, verbose)
        
        return pre_interval, case_ts, precision, recall, f1
    else:
        print("Threshold is invalid")