import re
import json
import argparse
import transformers 
from vllm import LLM, SamplingParams


def load_llm(base_model, tensor_parallel_size):
    llm = LLM(model=base_model, tensor_parallel_size=tensor_parallel_size)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model,
            padding_side="right",
            use_fast=False,
        )
    return llm, tokenizer

def get_k_shot_demo_str(demo_path, shots=3):
    with open(demo_path) as file:
        demos = json.load(file)

    input_str = "Below is an instruction that describes a task. \nWrite a response that appropriately completes the request.\n\n### Instruction:\n"
    for demo_data in demos[:shots]:
        output_steps = '\n'.join(demo_data['output_steps'])
        input_str += f"Act as a grade school math teacher and score the following problem solution.\nQuestion: {demo_data['question']}\n\nStudent Solution:\n{output_steps}"
        input_str +=f"\n\n### Response:\n{demo_data['evaluation']}"
        input_str +=f"\n\n### Instruction:\n"
    return input_str
    
def evaluate_vllm(
        llm,
        problem_instructions,
        temperature=1,
        top_p=0.9,
        max_new_tokens=2048,):
    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response", '</s>']
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop=stop_tokens)
    completions = llm.generate(problem_instructions, sampling_params)
    total_res = []
    for output in completions:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        total_res.append(generated_text)
    return total_res

def grade_diagGSM8k(to_be_graded_sols, input_str, llm):
    prompt_list = []
    for data in to_be_graded_sols:
        output_steps = '\n'.join(data['model_output_steps'])
        qa_pair_str = f"Act as a grade school math teacher and score the following problem solution.\nQuestion: {data['question']}\n\nStudent Solution:\n{output_steps}"
        input_prompt = input_str + qa_pair_str
        input_prompt +=f"\n\n### Response: Let's think step by step. \n"
        prompt_list.append(input_prompt)

    total_res = evaluate_vllm(llm, prompt_list, temperature=0.5, top_p=0.8, max_new_tokens=2048,)
    return total_res

def parse_results(to_be_graded_sols, total_res, model_prefix):
    graded_solutions = []
    for idx, response in enumerate(total_res):
        try:
            judgement = re.search(r"Final Judgement:\s*(.*?)\s*First Error Step:", response, re.DOTALL).group(1)
            error_step = re.search(r"First Error Step:\s*(.*?)\s*Error Analysis:", response, re.DOTALL).group(1)
            error_reason = re.search(r"Error Analysis:\s*(.*)", response, re.DOTALL).group(1)
            to_be_graded_sols[idx][f'{model_prefix}_eval_output'] = {
                "response": response,
                'correctness_pred': judgement,
                'error_step_pred': error_step,
                'error_reason': error_reason
            }
            graded_solutions.append(to_be_graded_sols[idx])
        except Exception as e:
            print(f"idx {idx} failed. Exception: {e} ")
            to_be_graded_sols[idx][f'{model_prefix}_eval_output'] = {
                "response": response,
                'correctness_pred': 'N/A',
                'error_step_pred': 'N/A',
                'error_reason': 'N/A'
            }
            graded_solutions.append(to_be_graded_sols[idx])
    return graded_solutions

        
        
        
def main():
    parser = argparse.ArgumentParser(
                    prog='EvalOpenSourceModels',
                    description='Script to reproduce the k-shot in context learning evaluations on open-source models')
    parser.add_argument('-k', '--k_shot_demos_path')
    parser.add_argument('-d', '--diagGSM8k_file_path')
    parser.add_argument('-o', '--output_dir_path')
    parser.add_argument('-m', '--model_path')
    parser.add_argument('-p', '--model_prefix', default='llama2')
    parser.add_argument('-t', '--tensor_parallel_size', default='1')
    args = parser.parse_args()    
    
    with open(args.diagGSM8k_file_path) as file:
        to_be_graded_sols = json.load(file)
    # initialize model
    llm, tokenizer = load_llm(args.model_path, args.tensor_parallel_size)
    # the k-shot-demos is uploaded in DiagGSM8k/dataset folder
    demo_str = get_k_shot_demo_str(args.k_shot_demos_path, shots=3)
    eval_res = grade_diagGSM8k(to_be_graded_sols, demo_str, llm)
    final_results = parse_results(to_be_graded_sols, eval_res, args.model_prefix)
    
    with open(f'{args.output_dir_path}/{args.model_prefix}_eval_results.json', 'w') as file:
        json.dump(final_results, file, indent=2, ensure_ascii=False)
