"""
    For single task evaluation, here we only evaluate RCL
"""

import os
from run_textual import log_pipeline, trace_pipeline
from utils.train import train
from utils.test import test
from model.AutoEncoder import AutoEncoder
from tqdm import tqdm
from agent.numerical_inference import Numerical_Inference
from agent.textual_inference import Textual_Inference
from agent.failure_diagnosis import Failure_Diagnosis
from utils.public_function import save_pkl, load_pkl
import re
import random
from model.tasks.evaluation import eval_RCL
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Hyper-parameters used in the model
batch_size = 8
num_nodes = 46
num_features = 110
learning_rate = 0.0001
weight_decay = 1e-4
num_epochs = 1000

model = AutoEncoder(num_nodes=num_nodes,num_features=num_features,tf_hidden_dim=num_features,gru_hidden_dim=64,num_heads=2,tf_layers=1,gru_layers=1)
model_path = "best_model.pth"
train(model, train_path="train_samples.pkl", batch_size = batch_size, num_nodes = num_nodes, num_features = num_features, learning_rate = learning_rate, weight_decay = weight_decay, num_epochs=num_epochs) 

# ######################################################
#                 Numerical Process                    #    
# ######################################################


result = test(model,model_path,train_path="train_samples.pkl", test_path="test_samples.pkl", gt_path="cases.pkl", mode=['FT'], split_ratio=0.7, before=300, after=300)
windows = [(entry["case_ts"], entry["case_ts"]) for entry in result] if result else []
numerical_answers = Numerical_Inference(mode=["RCL"], time_windows=windows,results=result)


######################################################
#               Textual Process FOR RCL              #    
######################################################

log_abstractions = []
trace_abstractions = []
for start_timestamp, end_timestamp in tqdm(windows, desc="Generating abstractions"):
    print(f"Start: {start_timestamp}, End: {end_timestamp}")
    
    log_abstraction = log_pipeline(start_time=start_timestamp-300, end_time=end_timestamp+300)
    trace_abstraction = trace_pipeline(start_time=start_timestamp-300, end_time=end_timestamp+300)
    
    log_abstractions.append(log_abstraction)
    trace_abstractions.append(trace_abstraction)

textual_answers = Textual_Inference(mode=["RCL"], time_windows=windows, log_abstractions=log_abstractions, trace_abstractions=trace_abstractions)

######################################################
#                 Extract Inputs FOR RCL             #    
######################################################
final_num_input=[]
final_tex_input=[]

# # # # --------------------Process numerical_answers ------------------------

for item in numerical_answers:
    services = []
    evidence = ""

    rank_match = re.search(r"<rcl_rank>\s*(.*?)\s*</rcl_rank>", item, re.DOTALL | re.IGNORECASE)
    if rank_match:
        rank_content = rank_match.group(1).strip()

        if rank_content.startswith("[") and rank_content.endswith("]"):
            rank_content = rank_content[1:-1]
        services = [s.strip().strip('"\'') for s in rank_content.split(",") if s.strip()]

    evidence_match = re.search(r"<Evidence>\s*(.*?)(?:</Evidence>|$)", item, re.DOTALL | re.IGNORECASE)
    if evidence_match:
        evidence = evidence_match.group(1).strip()

    if services:
        final_num_input.append({
            "Top5": services,
            "Evidence": evidence
        })
    else:
        final_num_input.append(
            item
        )


# ---------------------Process textual_answers------------------------

for item in textual_answers:
    services = []
    evidence = ""

    rank_match = re.search(r"<rcl_rank>\s*(.*?)\s*</rcl_rank>", item, re.DOTALL | re.IGNORECASE)
    if rank_match:
        rank_content = rank_match.group(1).strip()

        if rank_content.startswith("[") and rank_content.endswith("]"):
            rank_content = rank_content[1:-1]
        services = [s.strip().strip('"\'') for s in rank_content.split(",") if s.strip()]

    evidence_match = re.search(r"<Evidence>\s*(.*?)(?:</Evidence>|$)", item, re.DOTALL | re.IGNORECASE)
    if evidence_match:
        evidence = evidence_match.group(1).strip()

    if services:
        final_tex_input.append({
            "Top5": services,
            "Evidence": evidence
        })
    else:
        final_tex_input.append(
            item
        )




######################################################
#                 Failure Diagnosis                  #    
######################################################

final_answers = Failure_Diagnosis(mode=["RCL"], numerical_answers=final_num_input, textual_answers=final_tex_input)

######################################################
#                 RCL Top5 test                      #    
######################################################
pattern = re.compile(r'<rcl_rank>(.*?)</rcl_rank>', re.DOTALL)

# 解析函数
def parse_rcl_rank(content):
    content = content.strip()
    
    if content.startswith('[') and content.endswith(']'):
        content = content[1:-1]
    
    if content.startswith("'") or content.startswith('"'):
        try:
            return list(eval(content))
        except:
            pass
    return [item.strip() for item in content.split(',')]

answers = []
for item in numerical_answers:
    match = pattern.search(item)
    if match:
        extracted_content = match.group(1)
        parsed_list = parse_rcl_rank(extracted_content)
        for i in range(len(parsed_list)):
            item = parsed_list[i].lower()
            parsed_list[i] = item
        answers.append(parsed_list)
    else:
        print("In")
        answers.append(['None', 'None', 'None', 'None', 'None'])
service_dict = load_pkl("Your-service_dict-path")
for i in range(len(answers)):
    answer = answers[i]
    for item in answer:
        temp_answer = [service_dict.get(item, -1) for item in answer]
        new_answer = [random.randint(0, 45) if x == -1 else x for x in temp_answer] # random select if not found in service_dict
    answers[i] = new_answer

true_labels_list = [item["true_label"] for item in result]
accuracies = eval_RCL(answers, true_labels_list)
print(accuracies)
