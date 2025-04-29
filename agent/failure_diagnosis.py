from utils.public_function import llm_invoke_batch
from prompts.failure_expert import failure_system_prompt, ad_task_prompt,rcl_task_prompt, ft_task_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

def Failure_Diagnosis(mode, numerical_answers=None, textual_answers=None):
    user_prompts = []
    # Get user inputs for different tasks
    index = 1
    if "AD" in mode:
        for i in range(len(textual_answers)):
            inputs =  f"\nTask {index}:\n" + ad_task_prompt + f"\n Below is the Textual input:\n {textual_answers[i]}\n" + f"\n Below is the Numerical Expert output:\n " + str(numerical_answers[i])
            user_prompts.append(inputs)
        index += 1
    if "RCL" in mode:
        for i in range(len(numerical_answers)):
            inputs = f"\nTask {index}:\n" + rcl_task_prompt + f"\n Below is the Textual Expert output:\n {textual_answers[i]}\n"+ f"\n Below is the Numerical Expert output:\n " + str(numerical_answers[i])
            user_prompts.append(inputs)
        index += 1        
    if "FT" in mode:
        for i in range(len(numerical_answers)):
            inputs = f"\nTask {index}:\n" + ft_task_prompt + f"\n Below is the Textual input:\n {textual_answers[i]}\n"+ f"\n Below is the Numerical input:\n " + str(numerical_answers[i])
            user_prompts.append(inputs)
        index += 1        
    system_prompt = failure_system_prompt    


    # Get the model
    model_path = "Your-model-path"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype='auto',
        device_map='auto'
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    failure_answers = llm_invoke_batch(
        model, tokenizer, 
        system_prompt=system_prompt,
        user_prompts=user_prompts,
        max_token=512
    )
    return failure_answers