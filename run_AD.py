"""
    For single task evaluation, here we only evaluate AD
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
from model.tasks.evaluation import eval_AD
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

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
# #               Numerical Process                    #    
# ######################################################


result = test(model,model_path,train_path="train_samples.pkl", test_path="test_samples.pkl", gt_path="cases.pkl", mode=['AD'], split_ratio=0.7, before=300, after=300)
time_windows , case_ts = result
windows = time_windows if time_windows else []

numerical_answers = Numerical_Inference(mode=["AD"], time_windows=windows,results=None)

######################################################
#               Textual Process FOR  AD              #    
######################################################

log_abstractions = []
trace_abstractions = []

for time_window in tqdm(time_windows, desc="Generating abstractions"):
    start_timestamp = time_window[0] 
    end_timestamp = time_window[1]   
    log_abstraction = log_pipeline(start_time=start_timestamp, end_time=end_timestamp)
    trace_abstraction = trace_pipeline(start_time=start_timestamp, end_time=end_timestamp)

    log_abstractions.append(log_abstraction)
    trace_abstractions.append(trace_abstraction)

textual_answers = Textual_Inference(
    mode=["AD"],
    time_windows=time_windows,  
    log_abstractions=log_abstractions,
    trace_abstractions=trace_abstractions
)

######################################################
#                 Failure Diagnosis                  #    
######################################################

final_answers = Failure_Diagnosis(mode=["AD"], numerical_answers=numerical_answers, textual_answers=textual_answers)


######################################################
#                    AD  test                        #    
######################################################
pattern = re.compile(r'<ad_time>(?:\((.*?)\)|\[(.*?)\]|(.*?))</ad_time>')

def extract_timestamp(value):
    return int(re.search(r'\d+', value).group())

new_list = [
    [extract_timestamp(num.strip()) for num in (match.group(1) or match.group(2) or match.group(3)).split(',')]
    if (match := pattern.search(item)) else None
    for item in final_answers
]

pre_interval = []
for time in new_list:
    if time==None:
        continue
    s = time[0]
    e = time[1]
    if(e - s > 600):
        continue
    if s != e:
        pre_interval.append((s, e))

precision, recall, f1 = eval_AD(pre_interval, case_ts, impact_window=300, verbose=True)
print(f'precision = {precision}')
print(f'recall = {recall}')
print(f'f1 = {f1}')
