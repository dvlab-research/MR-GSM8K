import openai, time, re, json, argparse, os 

# TODO: Load api keys  
openai.api_key = os.environ.get('OPENAI_API_KEY', '')
openai.api_base = os.environ.get('OPENAI_API_BASE', '')

def request_helper(prompt, model_engine, max_retry=5):
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
                            temperature=0.5,
            )
            # 从响应中提取对话回复
            output = response["choices"][0]['message']['content'].strip()
            break
        except Exception as e:
            if retry < max_retry:
                print(f"Exception occurred, wait 15s: {e}", flush=True)
                time.sleep(15)
            else:
                output = ''
                break
            retry += 1
    return output



def request_apis(all_problems, graded_sols, output_file_path, model_engine):

    for idx, data in enumerate(all_problems[len(graded_sols):]):
        sol_steps = '\n'.join(data['model_output_steps'])
        prompt = f"""Act as a grade school math teacher and score the following problem solution.

Question:
{data['question']}

Student Solution:
{sol_steps}

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
        response = request_helper(prompt)
        judgement = re.search(r"Final Judgement:\s*(.*?)\s*First Error Step:", response, re.DOTALL).group(1)
        error_step = re.search(r"First Error Step:\s*(.*?)\s*Error Analysis:", response, re.DOTALL).group(1)
        error_reason = re.search(r"Error Analysis:\s*(.*)", response, re.DOTALL).group(1)
        data[f'{model_engine}_eval_output'] = {
            "response": response,
            'correctness_pred': judgement,
            'error_step_pred': error_step,
            'error_reason': error_reason
        }
        graded_sols.append(data)
        if (idx != 0 and idx % 20 ==0) or (idx+len(graded_sols) == len(all_problems)-1):
            with open(output_file_path, 'w') as file:
                json.dump(graded_sols, file, indent=2, ensure_ascii=False)
            print(f"Current progress: {len(graded_sols)} / {len(all_problems)}", flush=True)

def main():
    parser = argparse.ArgumentParser(
                    prog='EvalClosedSourceModels',
                    description='Script to reproduce the 0-shot evaluations on closed-source models via API')
    parser.add_argument('-d', '--diagGSM8k_file_path')
    parser.add_argument('-o', '--output_file_path')
    parser.add_argument('-p', '--model_engine', default='gpt-3.5-turbo')
    args = parser.parse_args() 

    with open(args.diagGSM8k_file_path) as file:
        all_problems = json.load(file)        
    try:
        with open(args.output_file_path) as file:
            graded_sols = json.load(file)
    except Exception as e:
        graded_sols = []
    request_apis(all_problems, graded_sols, args.output_file_path, args.model_engine)
