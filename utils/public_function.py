"""
    This file contains some frequently used function.
"""
import pickle
import torch
import dgl
import re
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from utils.spot import SPOT
from sklearn.feature_extraction.text import TfidfVectorizer
from drain3 import TemplateMiner


def softmax(x):
    """
    Softmax function to convert raw values into probabilities.
    """
    x = np.array(x) 
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def llm_invoke(model, tokenizer, system_prompt, user_prompt, max_token=512, temperature=1, verbose=False):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    if verbose:
        print(f'The number of input tokens is {model_inputs["input_ids"].numel()}')

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_token,
        temperature=temperature,
        do_sample=False 
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def llm_invoke_batch(model, tokenizer, system_prompt, user_prompts, max_token=512, temperature=1, batch_size=1, verbose=False):
    all_responses = []
    
    for i in tqdm(range(0, len(user_prompts), batch_size), desc="Processing batches"):
        batch_user_prompts = user_prompts[i:i+batch_size]
        batch_inputs = []

        for user_prompt in batch_user_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_inputs.append(text)

        model_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)

        if verbose:
            print(f'The number of input tokens is {model_inputs["input_ids"].numel()}')

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_token,
            temperature=temperature,
            do_sample=False  
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = [
            tokenizer.decode(output_ids, skip_special_tokens=True) for output_ids in generated_ids
        ]
        
        all_responses.extend(responses)

    return all_responses







#################################################################
#                   Functions For Numerical                     # 
#################################################################



def get_ild(deviation):
    ilds = []
    for i in range(deviation.shape[0]): 
        ilds.append(deviation[i])  
    return ilds


def get_sld(deviation):
    ilds = get_ild(deviation)
    l1_norms = [np.sum(np.abs(ild)) for ild in ilds]
    sorted_indices = np.argsort(l1_norms)[::-1]
    selected_indices = sorted_indices[:len(ilds)//2]
    selected_ilds = [ilds[i] for i in selected_indices]
    selected_l1_norms = [l1_norms[i] for i in selected_indices]
    weights = softmax(selected_l1_norms) 
    weighted_sum = np.zeros_like(selected_ilds[0])
    for i, ild in enumerate(selected_ilds):
        weighted_sum += weights[i] * ild
    return weighted_sum


def SLD(model, samples, top_k=3):
    from model.AutoEncoder import create_dataloader_AR
    mse = nn.MSELoss(reduction='none')
    system_level_deviation_df = pd.DataFrame()
    dataloader = create_dataloader_AR(samples, batch_size=64, shuffle=False)
    model.eval()
    with torch.no_grad():
        for timestamp, graphs, inputs, targets in dataloader:
            graphs = dgl.add_self_loop(graphs)
            outputs = model(graphs, inputs.float())
            loss = mse(outputs, targets) 

            # Get ILD
            instance_deviation = torch.sum(loss, dim=-1) 
            topk_values, topk_indices = torch.topk(instance_deviation, k=top_k, dim=-1) 
            mask = torch.zeros_like(instance_deviation)
            mask = mask.scatter_(1, topk_indices, 1).unsqueeze(-1) 
            
            # Get SLD
            system_level_deviation = torch.sum(loss * mask, dim=1) 

            tmp_df = pd.DataFrame(system_level_deviation.detach().numpy())
            tmp_df['timestamp'] = timestamp 
            
            # Concat SLD
            system_level_deviation_df = pd.concat([system_level_deviation_df, tmp_df])
    return system_level_deviation_df.reset_index(drop=True)


def ILD(model, test_samples):
    from model.AutoEncoder import create_dataloader_AR
    mse = nn.MSELoss(reduction='none')
    instance_level_deviation_df = pd.DataFrame()
    dataloader = create_dataloader_AR(test_samples, batch_size=64, shuffle=False)
    model.eval()
    with torch.no_grad():
         for timestamp, graphs, inputs, targets in tqdm(dataloader, desc="Generating ILDS"):
            graphs = dgl.add_self_loop(graphs)
            outputs = model(graphs, inputs.float())
            loss = mse(outputs, targets) 
            batch_size, instance_size, channel_size = loss.shape
            string_tensor = np.array([str(row.tolist()) for row in loss.reshape(-1, channel_size)]) 
            tmp_df = pd.DataFrame(string_tensor.reshape(batch_size, instance_size))
            tmp_df['timestamp'] = timestamp
            instance_level_deviation_df = pd.concat([instance_level_deviation_df, tmp_df])

    return instance_level_deviation_df.reset_index(drop=True)      

def aggregate_instance_representations(cases, instance_level_deviation_df, before=60, after=300):
    instance_representations = []
    for case in tqdm(cases,desc="Processing cases in aggregating ilds"):
        instance_representation = []
        timestamp = case[0]
        agg_df = instance_level_deviation_df[(instance_level_deviation_df['timestamp']>=(timestamp-before)) & (instance_level_deviation_df['timestamp']<timestamp+after)] 
        cnt = 0
        for col_name, col_data in agg_df.items():
            if col_name == 'timestamp':
                continue
            cnt += 1
            instance_representation.extend([(col_name, eval(item)) for item in col_data])
        instance_representations.append(instance_representation)
    return instance_representations
def aggregate_failure_representations(cases, system_level_deviation_df, type_hash=None, before=60, after=300):
    failure_representations, type_labels = [], []
    for case in tqdm(cases,desc="Processing cases in aggregating slds"):
        agg_df = system_level_deviation_df[(system_level_deviation_df['timestamp']>=(case[0]-before)) & (system_level_deviation_df['timestamp']<(case[0]+after))]
        failure_representations.append(list(agg_df.mean()[:-1])) 
        
        if type_hash:
            type_labels.append(case[1]) 
        else:
            type_labels.append(case[2])
    
    return failure_representations, type_labels








def run_spot(his_val, q=1e-2, level=0.98, verbose=False):
    model = SPOT(q)
    model.fit(his_val, [])
    model.initialize(level=level, verbose=verbose)
    return model.extreme_quantile

def get_threshold(his_val, q=1e-2, level=0.98, verbose=False):
    if len(set(his_val)) == 1:
        threshold = his_val[0]
    else:
        threshold = run_spot(his_val, q, level, verbose)
    return threshold












#################################################################
#                   Functions For Textual                       # 
#################################################################

def trace_filter(trace_data: pd.DataFrame, q = 0.95) -> pd.DataFrame:
    p95_values = trace_data.groupby("type")["duration"].quantile(q).to_dict()
    trace_data["is_high_latency"] = trace_data.apply(
        lambda row: row["duration"] > p95_values.get(row["type"], float('inf')), axis=1
    )
    high_latency_traces = trace_data[trace_data["is_high_latency"]]["trace_id"].unique()

    def get_full_trace(trace_id):
        spans = trace_data[trace_data["trace_id"] == trace_id]
        selected_spans = spans[spans["is_high_latency"]].copy()
        
        all_selected_spans = selected_spans.copy()
        while True:
            parents = trace_data[trace_data["span_id"].isin(all_selected_spans["parent_id"])]
            new_parents = parents[~parents["span_id"].isin(all_selected_spans["span_id"])]
            if new_parents.empty:
                break
            all_selected_spans = pd.concat([all_selected_spans, new_parents])
        
        return all_selected_spans

    result = pd.concat([get_full_trace(tid) for tid in high_latency_traces])

    return result



def log_filter(total_logs, keywords=["Failed", "hardware issue"], priority_quantile=0.95):
    template_miner = TemplateMiner()
    
    total_logs = total_logs.copy()
    
    def extract_template(log_message):
        try:
            result = template_miner.add_log_message(log_message)
            return result.get("log_template", log_message)
        except Exception:
            return log_message

    total_logs["template"] = total_logs["value"].fillna("").apply(extract_template)

    pattern = '|'.join(map(re.escape, keywords))
    keyword_mask = total_logs['value'].str.contains(pattern, case=False, na=False)
    filtered_by_keywords = total_logs[keyword_mask]
    
    if filtered_by_keywords.empty:
        return pd.DataFrame(columns=total_logs.columns)
    
    corpus = total_logs['template'].tolist()
    vectorizer = TfidfVectorizer(
        use_idf=True,
        binary=True,    
        smooth_idf=True,
        norm=None,       
        stop_words='english'
    )
    vectorizer.fit(corpus)
    
    filtered_corpus = filtered_by_keywords['template'].tolist()
    tfidf_matrix = vectorizer.transform(filtered_corpus)
    
    priority_scores = tfidf_matrix.mean(axis=1).A1
    filtered_by_keywords = filtered_by_keywords.copy()
    filtered_by_keywords['priority'] = priority_scores

    threshold = filtered_by_keywords['priority'].quantile(priority_quantile)
    filtered_logs = filtered_by_keywords[filtered_by_keywords['priority'] >= threshold]
    
    return filtered_logs.reset_index(drop=True)