"""
    For single task evaluation, here we only evaluate FT
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
import torch
import re
from model.tasks.evaluation import eval_FT
os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"

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

pre_type_names = result["pre_types"]
test_labels = [case[1] for case in result["target_cases"]]
case_ts = [case[0] for case in result["target_cases"]]
type_dict = load_pkl("Your-type_dict-path")


windows = [(entry, entry) for entry in case_ts] if result else []

numerical_answers = Numerical_Inference(mode=["FT"], time_windows=windows,results=pre_type_names)


######################################################
#                  Textual Process FT                #    
######################################################

log_abstractions = []
trace_abstractions = []
for start_timestamp, end_timestamp in tqdm(windows, desc="Generating abstractions"):
    print(f"Start: {start_timestamp}, End: {end_timestamp}")
    
    log_abstraction = log_pipeline(start_time=start_timestamp-300, end_time=end_timestamp+300)
    trace_abstraction = trace_pipeline(start_time=start_timestamp-300, end_time=end_timestamp+300)
    
    log_abstractions.append(log_abstraction)
    trace_abstractions.append(trace_abstraction)
textual_answers = Textual_Inference(mode=["FT"], time_windows=windows, log_abstractions=log_abstractions, trace_abstractions=trace_abstractions)

######################################################
#                 Extract Inputs FOR FT              #    
######################################################
final_num_input=[]
final_tex_input=[]

# ------------------------ Process numerical_answers ------------------------

for item in numerical_answers:
    services = []
    evidence = ""
    rank_match = re.search(r"<ft>\s*(.*?)\s*</ft>", item, re.DOTALL | re.IGNORECASE)
    if rank_match:
        rank_content = rank_match.group(1).strip()
    evidence_match = re.search(r"<Evidence>\s*(.*?)(?:</Evidence>|$)", item, re.DOTALL | re.IGNORECASE)
    if evidence_match:
        evidence = evidence_match.group(1).strip()

    if services:
        final_num_input.append({
            "Failure Type": rank_content,
            "Evidence": evidence
        })
    else:
        final_num_input.append(
            item
        )

# ------------------------Process textual_answers------------------------

final_tex_input = []

for item in textual_answers:
    services = []
    evidence = ""

    rank_match = re.search(r"<ft>\s*(.*?)\s*</ft>", item, re.DOTALL | re.IGNORECASE)
    if rank_match:
        rank_content = rank_match.group(1).strip()
    evidence_match = re.search(r"<Evidence>\s*(.*?)(?:</Evidence>|$)", item, re.DOTALL | re.IGNORECASE)
    if evidence_match:
        evidence = evidence_match.group(1).strip()
    if services:
        final_tex_input.append({
            "Top5": rank_content,
            "Evidence": evidence
        })
    else:
        final_tex_input.append(
            item
        )


#####################################################
#                Failure Diagnosis                  #    
#####################################################

final_answers = Failure_Diagnosis(mode=["FT"], numerical_answers=final_num_input, textual_answers=final_tex_input)




######################################################
#                      FT  test                      #    
######################################################
def extract_ft_content(data, default="Default String"):
    pattern = r'<ft>(.*?)</ft>'
    
    match = re.search(pattern, data)
    
    if match:
        content = match.group(1).strip()
        if content.startswith("[") and content.endswith("]"):
            try:
                parsed_list = eval(content)
                return parsed_list[0] if isinstance(parsed_list, list) and len(parsed_list) == 1 else default
            except:
                return content
        else:
            return content
    else:
        return default

data = []
for i in range(len(final_answers)):
    answer = final_answers[i]
    temp = extract_ft_content(answer, default="Container Hardware")
    data.append(temp)

reverse_type_dict = {v:k for k, v in type_dict.items()}
new_data = [reverse_type_dict.get(item, 0) for item in data]
for i in range(len(new_data)):
    print(f"{new_data[i]} --- {test_labels[i]}")
precision, recall, f1 = eval_FT(test_labels, new_data)
print(f"- Precision: {precision}")
print(f"- Recall: {recall}")
print(f"- F1 Score: {f1}")
