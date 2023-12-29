# direct asking 
import json, argparse
from vllm import LLM, SamplingParams

def evaluate_vllm(
    llm,
    instruction,
    use_cot=True,
    temperature=1,
    top_p=0.9,
    max_new_tokens=2048,):

    cot_problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    if type(instruction) == list:
        if use_cot == True:
            problem_instruction = [cot_problem_prompt.format(instruction=instruct) for instruct in instruction] 
        else:
            problem_instruction = [problem_prompt.format(instruction=instruct) for instruct in instruction] 
    else:
        if use_cot == True:
            prompt = cot_problem_prompt.format(instruction=instruction)
        else:
            prompt = problem_prompt.format(instruction=instruction)
        problem_instruction = [prompt]
    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response", '</s>']
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop=stop_tokens)
    completions = llm.generate(problem_instruction, sampling_params)
    total_res = []
    for output in completions:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        total_res.append(generated_text)
    return total_res
    

def get_prompt_list(to_be_graded_sols):
    prompt_list = []
    for data in to_be_graded_sols:
        output_steps = '\n'.join(data['model_output_steps'])
        prompt = f"""Act as a grade school math teacher and score the following problem solution.

Question:
{data['question']}

Student Solution:
{output_steps}

Your task involves three parts:
1. **Step-by-step Evaluation:** Go through the student solution carefully and identify key errors and potential misunderstandings that led to the incorrect solution.
2. **Final Judgement:**  Provide an overall judgement on the correctness of the student's solution.
3. **First Error Step:** If the solution is incorrect, generate the step number where the first error occurs, otherwise generate N/A here
4. **Error Analysis:** If the solution is incorrect, analyse the cause and reasons for the first error step, otherwise generate N/A here 

Here's the format I want:
Step-by-step Evaluation: [Provide a step by step examination of the student solution and identify key errors and misunderstandings here.]
Final Judgement: [Insert only **correct** or **wrong** here]
First Error Step: [Insert either N/A or the step number where the first error occurs]
Error Analysis: [Insert either N/A or the analysis of error in the first error among solution steps]

Please follow this format without any additional introductory or concluding statements.
"""
        prompt_list.append(prompt)
    return prompt_list


def main():
    parser = argparse.ArgumentParser(
                    prog='EvalLlama2',
                    description='Script to reproduce the 0-shot evaluations on fine-tuned llama2-70B models')
    parser.add_argument('-d', '--diagGSM8k_file_path')
    parser.add_argument('-o', '--output_dir_path')
    parser.add_argument('-m', '--model_path')
    parser.add_argument('-t', '--tensor_parallel_size', default='1')
    args = parser.parse_args()    
    
    with open(args.diagGSM8k_file_path) as file:
        to_be_graded_sols = json.load(file)
    
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size)
    prompt_list = get_prompt_list(to_be_graded_sols)
    total_res = evaluate_vllm(llm, prompt_list, use_cot=True, temperature=0.5, top_p=0.8, max_new_tokens=2048,)
    with open(args.output_dir_path, 'w') as file:
        json.dump(total_res, file, indent=2, ensure_ascii=False)