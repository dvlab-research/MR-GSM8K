import openai, time, re, json, argparse, os, math, tqdm
from concurrent.futures import ThreadPoolExecutor



def request_openai(client, prompt, model, max_retry=5, temperature=0., max_token=1024, stop_token_ids=None):
    message_dict = {
        "role": "user",
        "content": prompt
    }
    messages = []
    messages.append(message_dict)
    retry = 0
    extra_body = {"stop_token_ids": stop_token_ids} if stop_token_ids else None
    while True:
        try:
            completion = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_token,
                            temperature=temperature,
                            extra_body=extra_body
            )
            # 从响应中提取对话回复
            output = completion.choices[0].message.content
            break
        except Exception as e:
            if retry < max_retry:
                print(f"Exception occurred, wait 3s: {e}", flush=True)
                time.sleep(3)
            else:
                output = ''
                break
            retry += 1
    return output


def MCC_score(tp, tn, fp, fn):
    return (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

def get_mr_score(model_stat, w1=0.2, w2=0.3, w3=0.5):
    mcc_score = MCC_score(model_stat['t1-tp'], model_stat['t1-tn'], model_stat['t1-fp'], model_stat['t1-fn'])
    # mr_score_manual = w1 * max(0, mcc_score) + w2 * model_stat['t2-accuracy'] + w3 * model_stat['t3-accuracy']
    mr_score_auto = w1 * max(0, mcc_score) + w2 * model_stat['t2-accuracy'] + w3 * model_stat['t3-accuracy-auto']
    return  mcc_score, mr_score_auto

def construct_eval_stats(basic_stats, correct_sol_num=1427, incorrect_sol_num=1573):
    # true positive and true negative of task1-numbers that correctly determine the solution correctness
    t1_tp, t1_tn = basic_stats['t1-tp'], basic_stats['t1-tn']
    # correct number of task 2-determine the first error step
    t2_corr_num = basic_stats['t2_corr_num']
    # correct number of task 3-determine the error reason, judged either by annotators or GPT4
    # t3_corr_num_human, t3_corr_num_auto = basic_stats['t3_corr_num_human'], basic_stats['t3_corr_num_auto']
    t3_corr_num_auto = basic_stats['t3_corr_num_auto']
    final_stats = {
            't1-tp': t1_tp,
            't1-tn': t1_tn,
            't1-fp': (incorrect_sol_num-t1_tn),
            't1-fn': (correct_sol_num-t1_tp),
            't1-recall': t1_tp/correct_sol_num,
            't1-precision': t1_tp/(t1_tp+incorrect_sol_num-t1_tn), # precision = tp/(tp+fp)
            't1-fpr': t1_tn/incorrect_sol_num,
            't2-accuracy': t2_corr_num/incorrect_sol_num,
            # 't3-accuracy': t3_corr_num_human/incorrect_sol_num,
            't3-accuracy-auto': t3_corr_num_auto/incorrect_sol_num,
    }
    return final_stats

def process_eval_results(grade_responses):
    task1_true_positive, task1_true_negative = 0, 0
    task2_accy, task3_accy_human = 0, 0
    task3_accy_auto = 0
    step_mapper = {f"step {i}": f"{i}"  for i in range(30)}
    for data in grade_responses:
        if data['Evaluation_Result']['solution_correctness'].strip().lower() == 'incorrect':
            correctness_pred = 'wrong'
        else:
            correctness_pred = data['Evaluation_Result']['solution_correctness'].strip().lower()
        if data['model_output_solution_correctness'].lower() in correctness_pred:
            if data['model_output_solution_correctness'].lower() == 'correct':
                task1_true_positive +=1
            else:
                task1_true_negative +=1
                # only if the solution is incorrect and the model agrees on the incorrectness do 
                # we look into task2 and task3 performance
                if data['Evaluation_Result']['first_error_step'].strip().isdigit():
                    error_step_pred = data['Evaluation_Result']['first_error_step'].strip()
                elif data['Evaluation_Result']['first_error_step'].strip().lower() in step_mapper:
                    error_step_pred = step_mapper[data['Evaluation_Result']['first_error_step'].strip().lower()]
                else:
                    error_step_pred = ''
                if error_step_pred == str(data['model_output_solution_first_error_step']):
                    task2_accy += 1
                    if 'correct' in data['Error_Reason_Correctness_Analysis']['error_reason_correctness'].lower():
                        task3_accy_auto +=1
    return {
        't1-tp': task1_true_positive,
        't1-tn': task1_true_negative,
        't2_corr_num': task2_accy,
        't3_corr_num_auto': task3_accy_auto
    }
           
def construct_k_shot_demos(demo_path, num_shots):
    if num_shots == 0 or not demo_path:
        return ''
    with open(demo_path) as file:
        k_shot_demos = json.load(file)[:num_shots]
    demo_str = 'Here is some demo examples: '
    for demo in k_shot_demos:
        sol_steps = '\n'.join(demo['output_steps'])
        demo_str += f"\n\nQuestion:\n{demo['question']}\n\nStudent Solution:\n{sol_steps}\nStep-by-step Evaluation:\n{demo['evaluation']}"    
    return demo_str

def single_thread_eval_generation(data, client, model_name, stop_token_ids):
    prompt = data.pop('eval_prompt')
    response = request_openai(client, prompt, model_name, stop_token_ids=stop_token_ids)
    try:
        judgement = re.search(r"Final Judgement:\s*(.*?)\s*First Error Step:", response, re.DOTALL).group(1)
        error_step = re.search(r"First Error Step:\s*(.*?)\s*Error Analysis:", response, re.DOTALL).group(1)
        error_reason = re.search(r"Error Analysis:\s*(.*)", response, re.DOTALL).group(1)
    except Exception as e:
        judgement = error_step = error_reason = 'PARSE_ERROR'
    data[f'Evaluation_Result'] = {
        'evaluated_model': model_name,
        'evaluation_prompt': prompt,
        "evaluation_raw_response": response,
        'solution_correctness': judgement,
        'first_error_step': error_step,
        'error_reason': error_reason
    }
    

def single_thread_grade_generation(data, client, model_name):
    prompt = data.pop('grade_prompt')
    response = request_openai(client, prompt, model_name)
    try:
        reasoning = re.search(r"Step-by-Step Reasoning:\s*(.*?)\s*Student Error Reason Analysis:", response, re.DOTALL).group(1)
        error_analysis = re.search(r"Student Error Reason Analysis:\s*(.*?)\s*Final Decision:", response, re.DOTALL).group(1)
        final_decision = re.search(r"Final Decision:\s*(.*)", response, re.DOTALL).group(1)
    except Exception as e:
        reasoning = error_analysis = final_decision = 'PARSE_ERROR'
    data['Error_Reason_Correctness_Analysis'] = {
        'scoring_model': model_name,
        'scoring_prompt': prompt, 
        'scoring_raw_response': response,
        'annotation_analysis': reasoning,
        'error_reason_analysis': error_analysis,
        'error_reason_correctness': final_decision
    }

    
def multi_thread_response_generation(data_list, client, model_name, stop_token_ids, max_workers=5, type='eval'):
    if type == 'eval':
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(
                tqdm.tqdm(
                    executor.map(lambda x: single_thread_eval_generation(x, client, model_name, stop_token_ids), data_list),
                    total=len(data_list)
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(
                tqdm.tqdm(
                    executor.map(lambda x: single_thread_grade_generation(x, client, model_name), data_list),
                    total=len(data_list)
                )
            )
        

def grade_eval_outputs(eval_responses, output_file_path, client, model_name, max_workers):
    data_list = []
    step_mapper = {f"step {i}": f"{i}"  for i in range(30)}
    for data in eval_responses:
        data['Need_Error_Reason_Review'] = False
        if data['Evaluation_Result']['solution_correctness'].strip().lower() == 'incorrect':
            correctness_pred = 'wrong'
        else:
            correctness_pred = data['Evaluation_Result']['solution_correctness'].strip().lower()
        if data['model_output_solution_correctness'].lower() in correctness_pred:
            if data['model_output_solution_correctness'].lower() == 'wrong':
                # only if the solution is incorrect and the model agrees on the incorrectness do 
                # we look into task2 and task3 performance
                if data['Evaluation_Result']['first_error_step'].strip().isdigit():
                    error_step_pred = data['Evaluation_Result']['first_error_step'].strip()
                elif data['Evaluation_Result']['first_error_step'].strip().lower() in step_mapper:
                    error_step_pred = step_mapper[data['Evaluation_Result']['first_error_step'].strip().lower()]
                else:
                    error_step_pred = ''
                if error_step_pred == str(data['model_output_solution_first_error_step']):
                    data['Need_Error_Reason_Review'] = True
    for data in eval_responses:
        if data['Need_Error_Reason_Review']:
            if "Error_Reason_Correctness_Analysis" in data and data["Error_Reason_Correctness_Analysis"]["scoring_raw_response"]: continue
            data['grade_prompt'] = f"""Hello GPT-4,
As an experienced grade-school math teacher, your assistance is required to evaluate a student's explanation regarding the error in a math problem solution. The task involves a detailed understanding of the math problem, the incorrect solution provided, and the ground truth behind the error. Your analysis should focus on whether the student's explanation aligns with the actual error in the solution.

Please find the details below:

- Math Question: {data['question']}
- Incorrect Solution Provided: {data['model_output_steps']}
- First Incorrect Step in the Solution: {data['model_output_solution_first_error_step']}
- Ground Truth Error Reason: {data['model_output_solution_first_error_reason']}
- Student's Explanation of the Error: {data['Evaluation_Result']['error_reason']}

Based on this information, please provide the following:

1. Step-by-Step Reasoning: [Offer a succinct, step-by-step interpretation of the ground truth error reason.]
2. Student Error Reason Analysis: [Analyze the student's explanation step by step, determining its accuracy in reflecting the actual error briefly.]
3. Final Decision: [State only 'Correct' or 'Wrong' to conclude whether the student's explanation correctly identifies the error based on your analysis.]

Please follow this format without any additional introductory or concluding statements."""
            data_list.append(data)

    multi_thread_response_generation(data_list, client, model_name, None, max_workers, type='grade')
    with open(output_file_path, 'w') as file:
        json.dump(eval_responses, file, indent=2, ensure_ascii=False)
    return eval_responses
    


def get_eval_outputs(all_problems, k_shot_demos, output_file_path, client, model_name, stop_token_ids, max_workers=5):
    data_list = []
    for data in all_problems:
        sol_steps = '\n'.join(data['model_output_steps'])
        if not k_shot_demos:
            data['eval_prompt'] = f"""Act as a grade school math teacher and score the following problem solution.

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
        else:
            data['eval_prompt'] = f"""Act as a grade school math teacher and score the following problem solution.
            
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

{k_shot_demos}

Here is the question and solution for you to solve, please follow the desired format without any additional introductory or concluding statements
Question:
{data['question']}

Student Solution:
{sol_steps}
"""
        if "Evaluation_Result" in data and data["Evaluation_Result"]["evaluation_raw_response"]: continue
        data_list.append(data)
    multi_thread_response_generation(data_list, client, model_name, stop_token_ids, max_workers, type='eval')
    with open(output_file_path, 'w') as file:
        json.dump(all_problems, file, indent=2, ensure_ascii=False)
    return all_problems
    

def main():
    parser = argparse.ArgumentParser(prog='EvalMRGSM8K', description='Script to evaluate MR-GSM8K dataset')
    parser.add_argument('-d', '--diagGSM8k_file_path')
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('--score_base_url', default='', help='the base_url for scoring the responses from evaluated model')
    parser.add_argument('--score_api_key', default='', help='the api_key for scoring the responses from evaluated model')
    parser.add_argument('--score_model_name', default='gpt-4-turbo', help='the model name for scoring the responses from evaluated model')
    parser.add_argument('--eval_base_url', default='http://0.0.0.0:8888/v1', help='the base_url for evaluated model ')
    parser.add_argument('--eval_api_key', default='placeholder', help='the api_key for evaluated model')
    parser.add_argument('--eval_model_name', default='/path/to/your/model', help='the model name for evaluated model, if you are using open source models served from local vllm server, this is the absolute path your model')
    parser.add_argument('--stop_token_ids', type=int, required=False,  nargs="+", help='List of stop token ids because default tokenizer used by vllm might not using correct stop tokens in chat models.')
    parser.add_argument('--max_workers', type=int, default=5, required=False, help='Number of parallel workers for API requests.')
    parser.add_argument('--shot_num', '-k', type=int, required=False, default=0, help='The number of demonstrations for evaluated model')
    parser.add_argument('--demo_path', type=str, required=False, default='', help='The path to the k-shot-demos.json file')
    
    args = parser.parse_args()
    with open(args.diagGSM8k_file_path) as file:
        all_problems = json.load(file)
    if '/' in args.eval_model_name:
        # name of open-sourced model served by vllm is the absolute path of the downloaded model folder
        succint_model_name = args.eval_model_name.split('/')[-1]
    else:
        succint_model_name = args.eval_model_name # commercial models
    k_shot_demos = construct_k_shot_demos(args.demo_path, args.shot_num)
    eval_output_path = f"{args.output_dir}/{succint_model_name}_{args.shot_num}shot_eval_results.json"
    scored_output_path = f"{args.output_dir}/{succint_model_name}_{args.shot_num}shot_scored_eval_results.json"
    # Evaluate on MR-GSM8k
    if os.path.exists(eval_output_path):
        with open(eval_output_path) as file:
            all_problems = json.load(file)    
    eval_client = openai.OpenAI(base_url=args.eval_base_url, api_key=args.eval_api_key)
    eval_responses = get_eval_outputs(all_problems, 
                                        k_shot_demos,
                                        eval_output_path, 
                                        eval_client, 
                                        args.eval_model_name, 
                                        args.stop_token_ids, 
                                        args.max_workers)

    # Use LLMs to grade error reasons
    if os.path.exists(scored_output_path):
        with open(scored_output_path) as file:
            eval_responses = json.load(file)
    score_client = openai.OpenAI(base_url=args.score_base_url, api_key=args.score_api_key)
    grade_responses = grade_eval_outputs(eval_responses, 
                                        scored_output_path, 
                                        score_client, 
                                        args.score_model_name, 
                                        args.max_workers)
    # Calculate MR-Scores
    eval_stats = construct_eval_stats(process_eval_results(grade_responses))
    mcc_score, mr_score = get_mr_score(eval_stats)
    print(f"Task1 True Positive Rate: {round(eval_stats['t1-recall']*100, 1)}")
    print(f"Task1 True Negative Rate: {round(eval_stats['t1-fpr']*100, 1)}")
    print(f"Task1 MCC Score: {round(mcc_score*100, 1)}")
    print(f"Task2 Accuracy: {round(eval_stats['t2-accuracy']*100, 1)}")
    print(f"Task3 Accuracy: {round(eval_stats['t3-accuracy-auto']*100, 1)}")
    print(f"MR-Score: {round(mr_score*100, 1)}")
    
if __name__ == "__main__":
    main()
