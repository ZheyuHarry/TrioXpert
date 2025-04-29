from utils.public_function import llm_invoke_batch
from prompts.textual_expert import textual_system_prompt, ad_task_prompt,rcl_task_prompt, ft_task_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

def Textual_Inference(mode=["AD"], time_windows=None, log_abstractions=None, trace_abstractions=None):
    user_prompts = []
    index = 1
    if "AD" in mode:
        for i in range(len(time_windows)):
            inputs = ad_task_prompt + f"\n Below is the log abstraction:\n {log_abstractions[i]}\n" + f"\n Below is the trace abstraction:\n {trace_abstractions[i]}\n" f"Task {index}:\n"
            user_prompts.append(inputs)
        index += 1
    system_prompt = textual_system_prompt
    if "RCL"in mode:
        for i in range(len(time_windows)):
            inputs =  rcl_task_prompt +  f"\n Below is the log abstraction:\n {log_abstractions[i]}\n" + f"\n Below is the trace abstraction:\n {trace_abstractions[i]}\n" f"Task {index}:\n"
            user_prompts.append(inputs)
        index += 1
    if "FT"in mode:
        for i in range(len(time_windows)):
            inputs = ft_task_prompt +  f"\n Below is the log abstraction:\n {log_abstractions[i]}\n" + f"\n Below is the trace abstraction:\n {trace_abstractions[i]}\n" f"Task {index}:\n"
            user_prompts.append(inputs)
        index += 1

    # Get the model
    model_path = "Your-model-path"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype='auto',
        device_map='auto'
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    textual_answers = llm_invoke_batch(
        model, tokenizer, 
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        max_token=512
    )
    return textual_answers