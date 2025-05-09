# Getting Started

## Environment
Python 3.10.16, PyTorch 2.4.0, Transformers 4.46.0, Accelerate 1.4.0, and DGL 2.4.0+cu124 are suggested.

## Dataset
Dataset D1 is derived from a simulated e-commerce system based on a microservice architecture, which is deployed on a top bank’s cloud platform. Its traffic patterns are consistent with real-world business scenarios, and the failure types are summarized from common issues observed in real systems. The dataset includes 40 microservice instances and 6 virtual machines, covering five distinct failure types. The failure scenarios are derived from actual failures (Container Hardware, Container Network, Node CPU, Node Disk, and Node Memory-related failures). The collected records were labeled with their respective root cause instances and failure types. The raw data and labels of D1 are publicly available at https://drive.google.com/file/d/1T3mQgwBrICIQOKjbKnptgQNphPoD2w1W/view?usp=sharing.

Dataset D2 is sourced from the management system of a top-tier commercial bank and encompasses real-world business application scenarios. The dataset comprises a total of 18 instances, involving various components such as microservices, servers, databases, and Docker, and is categorized into six failure types. D2 has been used in the International AIOps Challenge 2021 https://aiops-challenge.com. Due to the non-disclosure agreement, we cannot make it publicly available. 


In the folder:
- `cases`: There is one file in this directory.   
  cases.pkl: The three items in the table header indicate the failure injection timestamp, root cause of the failure, and failure type respectively. 

- `hash_info`: There are four files in this directory. They all hold a dictionary that records the correspondence between names and indexes.

- `samples`: There are two files in this directory, samples for pre-training (train_samples.pkl), and samples for evaluation (test_samples.pkl). 

Each sample is a tuple: (timestamp, graphs, features of each node). Graphs indicate the topology of the microservice system generated from call relationships and deployment information; Features of each node are composed of pod metric feats, pod trace feats, pod log feats and node metric feats. 

## Demo
We provide a demo.
And you should specify the data_path and model_path in the files.
Before running the following commands, please unzip D1.zip.

Let's take RCL task as an example:

```
python run_RCL.py
```

## Parameter Description in the Demo

Take dataset D1 as an example.

### Pre-training in data process pipeline 1.

* `instance_dim`: The number of microservice instances. (default: 46)
* `num_heads`: The number of attention heads in a Transformer Encoder. (default: 2)
* `tf_layers`: The number of layers of Transformer Encoder. (default: 1)
* `channel_dim`: The number of data channels. (default: 110)
* `gnn_hidden_dim`: The hidden dimension of GraphSAGE. (default: 64)
* `gnn_out_dim`: The output dimension of GraphSAGE. (default: 32)
* `gnn_layers`: The number of layers of GraphSAGE. (default: 2)
* `gru_hidden_dim`: The hidden dimension of GRU. (default: 32)
* `gru_layers`: The number of layers of GRU. (default: 1)
* `epochs`: The training epochs. (default: 1000)
* `batch_size`: The batch size. (default: 8)
* `learning_rate`: The learning rate. (default: 0.0001)

### Downstream tasks in data process pipeline 1.

* `split_ratio`: The ratio for splitting the initialization set and the test set. (default: 0.7)
* `q & level`: The initialize parameters for SPOT. (default: 0.21, 0.98)
* `delay`: Consider anomalies with intervals smaller than the delay as the same failure. (default: 600)
* `impact_window`: The impact range of a single failure. Functions similarly to 'before' and 'after'. (default: 300)
* `before & after`: The failure injection time is inject_ts. Assuming the failure impact range is [inject_ts-before, inject_ts+after], take the data within this time window for analysis. (default: 300, 300)
* `max_clusters`: The maximum number of clusters obtained from the cut tree in failure triage. (default: 50)
* `verbose`: Control the output to be either concise or verbose. (default: False)

### Filtering in data process pipeline 2 & 3.
* `q`: The quantile parameter in the trace filter used to identify high-latency spans. (default: 0.95)
* `priority_quantile`: The quantile parameter in the log filter used to identify distinct log entries. (default: 0.95)
* `keywords`: The parameter in the log filter used to identify failure-related log entries. (default: [`failure`, `exception`, `error`, `debug`])

### Large language model invocation.
* `model_path`: The name specifying the path for loading the large language model. (dafault: `Qwen2.5-7B-Instruct`)
* `max_token`: The parameter defines the maximum number of tokens allowed in text generation. (default: 512)
