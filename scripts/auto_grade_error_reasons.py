import json, os, argparse
import openai, time, re, json
from tqdm import tqdm

# TODO: Load api keys  
openai.api_key = os.environ.get('OPENAI_API_KEY', '')
openai.api_base = os.environ.get('OPENAI_API_BASE', '')


def request_openai(prompt, model_engine = 'gpt-4-1106-preview', max_retry=5):
    message_dict = {
        "role": "user",
        "content": prompt
    }
    messages = []
    messages.append(message_dict)
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                            model=model_engine,
                            messages=messages,
                            max_tokens=2048,
                            temperature=.5,
            )
            # 从响应中提取对话回复
            output = response["choices"][0]['message']['content'].strip()
            break
        except Exception as e:
            if retry < max_retry:
                print(f"Exception '{e}' occurred, wait 15s", flush=True)
                time.sleep(15)
            else:
                output = ''
                break
            retry += 1
    return output

def main():
    parser = argparse.ArgumentParser(
                    prog='EvalErrorReasons',
                    description='Script to use GPT4 as our error reason labeller')
    parser.add_argument('-d', '--eval_results_path')
    parser.add_argument('-o', '--output_dir_path')
    parser.add_argument('-p', '--model_engine', default='gpt-3.5-turbo')
    args = parser.parse_args() 
    mapper = {
        "wizardmath_70B_eval_results.json": 'wizardmath_70B_eval_output',
        "Claude_eval_results.json": 'claude2_eval_output',
        "GPT3_5_eval_results.json": 'gpt3_5_eval_output',
        "GPT4_eval_results.json": 'gpt4_eval_output',
        "llama2_70b_eval_results.json": 'llama2_eval_output',
        "mammoth_70B_eval_results.json": 'mammoth_70B_eval_output',
        "metamath_70B_eval_results.json": 'metamath_70B_eval_output'
    }
    for entry in os.scandir(args.eval_results_path):
        if entry.is_file() and entry.name.endswith('.json'):
            with open(entry.path) as file:
                eval_data = json.load(file)
                to_be_graded_data = [data for data in eval_data if data['error_reason_correctness'] != 'N/A']
                try:
                    with open(f'{args.output_dir_path}/{entry.name}') as file:
                        grade_results = json.load(file)
                except Exception as e:
                    grade_results = []
                for data in tqdm(to_be_graded_data[len(grade_results):]):
                    eval_field = mapper[entry.name]
                    prompt = f"""Hello GPT-4,
As an experienced grade-school math teacher, your assistance is required to evaluate a student's explanation regarding the error in a math problem solution. The task involves a detailed understanding of the math problem, the incorrect solution provided, and the ground truth behind the error. Your analysis should focus on whether the student's explanation aligns with the actual error in the solution.

Please find the details below:

- Math Question: {data['question']}
- Incorrect Solution Provided: {data['model_output_steps']}
- First Incorrect Step in the Solution: {data['model_output_solution_first_error_step']}
- Ground Truth Error Reason: {data['model_output_solution_first_error_reason']}
- Student's Explanation of the Error: {data[eval_field]['error_reason']}

Based on this information, please provide the following:

1. Step-by-Step Reasoning: [Offer a succinct, step-by-step interpretation of the ground truth error reason.]
2. Student Error Reason Analysis: [Analyze the student's explanation step by step, determining its accuracy in reflecting the actual error briefly.]
3. Final Decision: [State only 'Correct' or 'Wrong' to conclude whether the student's explanation correctly identifies the error based on your analysis.]

Please follow this format without any additional introductory or concluding statements."""
                    response = request_openai(prompt)
                    try:
                        reasoning = re.search(r"Step-by-Step Reasoning:\s*(.*?)\s*Student Error Reason Analysis:", response, re.DOTALL).group(1)
                        error_analysis = re.search(r"Student Error Reason Analysis:\s*(.*?)\s*Final Decision:", response, re.DOTALL).group(1)
                        final_decision = re.search(r"Final Decision:\s*(.*)", response, re.DOTALL).group(1)
                        data['gpt4_error_reason_correctness_analysis'] = {
                            'response': response,
                            'Error Comprehension': reasoning,
                            'Model Error Reason Analysis': error_analysis,
                            'Final Decision': final_decision
                        }
                    except Exception as e:
                        data['gpt4_error_reason_correctness_analysis'] = {
                            'response': response,
                            'Error Comprehension': 'ERROR_PARSING',
                            'Model Error Reason Analysis': 'ERROR_PARSING',
                            'Final Decision': 'ERROR_PARSING'
                        }
                    grade_results.append(data)
                    with open(f'{args.output_dir_path}/{entry.name}', 'w') as file:
                        json.dump(grade_results, file, indent=2)