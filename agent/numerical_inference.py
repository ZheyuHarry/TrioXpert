from utils.public_function import llm_invoke_batch
from prompts.numerical_expert import numerical_system_prompt,  ad_task_prompt,rcl_task_prompt, ft_task_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

def Numerical_Inference(mode=["AD"], time_windows=None, results=None):
    user_prompts = []
    index = 1

    if "AD" in mode:
        for time_window in time_windows:
            inputs = (
                f"\n\n## Task {index}\n" 
                + ad_task_prompt 
                + f"({time_window[0], time_window[1]})\n"
            )
            user_prompts.append(inputs)
        index += 1

    elif "RCL" in mode:
        flat_results = []
        for item in results:
            if isinstance(item, list):
                flat_results.extend(item)
            else:
                flat_results.append(item)
        

        for case_idx, result in enumerate(flat_results, start=1):
            if not isinstance(result, dict):
                continue  

            pred_rc_names = result.get('pred_rc_names', [])
            case_info = (
                f"- Predicted Top 5 Names: {', '.join(pred_rc_names)}\n"
            )
            task_prompt = (
        
                f"\n\n## Task {index}.{case_idx}\n" 
                + rcl_task_prompt 
                + case_info
            )
            user_prompts.append(task_prompt)
        
        index += 1

    elif "FT" in mode:
        for pre_type_name in results:
            task_prompt = (
                f"\n\n## Task {index}\n" 
                + ft_task_prompt 
                + pre_type_name
            )
            user_prompts.append(task_prompt)

    system_prompt = numerical_system_prompt

    model_path = "Your-model-path"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype='auto',
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    numerical_answers = llm_invoke_batch(
        model, tokenizer, 
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        max_token=512
    )
    return numerical_answers
    
