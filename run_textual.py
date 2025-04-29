"""
    This file is used to run the process of sketching the microservice system in the form of matrix
"""

import argparse
import os
from datetime import datetime
import pandas as pd
import pytz
from utils.public_function import llm_invoke, log_filter, trace_filter, load_pkl, save_pkl
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts.log_pipeline import log_abstract_prompt, log_abstract_system_prompt
from prompts.trace_pipeline import trace_abstract_prompt, trace_abstract_system_prompt

os.environ["CUDA_VISIBLE_DEVICES"] = "1，2，4"



def log_pipeline(start_time, end_time, mode = ['AD', 'FT', 'RCL']):
    """ 
        Get the raw logs -> total_logs
    """
    log_path = 'Your-log-path'
    log_data = load_pkl(log_path)
    log_data = log_data[(log_data['timestamp'] >= start_time) & (log_data['timestamp'] <= end_time)]
    
    log_data = log_data[['timestamp', 'cmdb_id', 'value']]

    total_logs = pd.DataFrame()
    total_logs = pd.concat([total_logs, log_data], ignore_index=True)
    
    print(f'Length of total logs are {len(total_logs)}')


    """
        total_logs -> filter out important logs -> filtered_logs
    """
    keywords = ["failure","Exception", "error","debug"]
    filtered_logs = log_filter(total_logs, keywords)
    print(len(filtered_logs))
    # print(filtered_logs)

    filtered_logs = filtered_logs.sort_values(by="timestamp", ascending=True)

    """
        filtered_logs -> generate the abstract based on logs -> log_abstract
    """
    log_inputs = ""
    for index, row in filtered_logs.iterrows():
        timestamp = row["timestamp"]
        service = row["cmdb_id"]
        value = row["value"]
        log_inputs = log_inputs + f"- [Timestamp: {timestamp}], service \'{service}\': {value}.\n"
    
    system_prompt = log_abstract_system_prompt
    user_prompt = log_abstract_prompt + log_inputs

    # Get the model
    model_path = "Your-model-path"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype='auto',
        device_map='auto'
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    log_abstract = llm_invoke(
        model, tokenizer, 
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_token=512
    )
    return log_abstract




def trace_pipeline(start_time, end_time, mode = ['AD', 'FT', 'RCL']):
    """ 
        Get the raw traces -> trace_data
    """
    
    # get trace data
    trace_path = f"Your-trace-Path"
    trace_data = pd.read_csv(trace_path)

    # filter the timestamp
    trace_data = trace_data[(trace_data["timestamp"]//1000 >= start_time) & (trace_data["timestamp"]//1000 <= end_time)]
    print(f"Length of total traces are {len(trace_data)}")


    """
        trace_data -> filter out important traces -> filtered_traces
    """
    filtered_traces = trace_filter(trace_data, q = 0.99)
    # print(filtered_traces.head(15))
    print(f"Length of the filtered traces are {len(filtered_traces)}")

    """
        filtered_traces -> Form serielized inputs for LLM -> trace_inputs
    """
    parent_to_caller = {row["span_id"]: row["cmdb_id"] for _, row in filtered_traces.iterrows()}

    trace_dict = {}
    for trace_id, group in filtered_traces.groupby("trace_id"):
        group = group.sort_values("timestamp")
        
        formatted_spans = []
        for _, row in group.iterrows():
            caller = parent_to_caller.get(row["parent_id"], "ROOT") 
            callee = row["cmdb_id"]
            span_tuple = (row["timestamp"], caller, callee, row["duration"], row["is_high_latency"], row["type"])
            formatted_spans.append(span_tuple)
        
        trace_dict[trace_id] = formatted_spans

    trace_inputs = ""
    for trace_id, spans in trace_dict.items():
        trace_inputs += f"<{trace_id}>\n"
        for span in spans:
            trace_inputs += f"{span},\n"
        trace_inputs += f"</{trace_id}>\n\n"

    """
        trace_inputs -> generate the abstract based on traces -> trace_abstract
    """
    system_prompt = trace_abstract_system_prompt
    user_prompt = trace_abstract_prompt + trace_inputs

    # Get the model
    model_path = "Your-model-path"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype='auto',
        device_map='auto'
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    trace_abstract = llm_invoke(
        model, tokenizer, 
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_token=512
    )
    return trace_abstract

