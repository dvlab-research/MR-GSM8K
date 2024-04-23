import os
import json 
import math

def MCC_score(tp, tn, fp, fn):
    return (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

def mr_score(model_stat, w1=0.2, w2=0.3, w3=0.5):
    mcc_score = MCC_score(model_stat['t1-tp'], model_stat['t1-tn'], model_stat['t1-fp'], model_stat['t1-fn'])
    mr_score_manual = w1 * max(0, mcc_score) + w2 * model_stat['t2-accuracy'] + w3 * model_stat['t3-accuracy']
    mr_score_auto = w1 * max(0, mcc_score) + w2 * model_stat['t2-accuracy'] + w3 * model_stat['t3-accuracy-auto']
    return mr_score_manual, mr_score_auto

def construct_eval_stats(basic_stats, correct_sol_num=1427, incorrect_sol_num=1573):
    # true positive and true negative of task1-numbers that correctly determine the solution correctness
    t1_tp, t1_tn = basic_stats['t1-tp'], basic_stats['t1-tn']
    # correct number of task 2-determine the first error step
    t2_corr_num = basic_stats['t2_corr_num']
    # correct number of task 3-determine the error reason, judged either by annotators or GPT4
    t3_corr_num_human, t3_corr_num_auto = basic_stats['t3_corr_num_human'], basic_stats['t3_corr_num_auto']
    final_stats = {
            't1-tp': t1_tp,
            't1-tn': t1_tn,
            't1-fp': (incorrect_sol_num-t1_tn),
            't1-fn': (correct_sol_num-t1_tp),
            't1-recall': t1_tp/correct_sol_num,
            't1-precision': t1_tp/(t1_tp+incorrect_sol_num-t1_tn), # precision = tp/(tp+fp)
            't2-accuracy': t2_corr_num/incorrect_sol_num,
            't3-accuracy': t3_corr_num_human/incorrect_sol_num,
            't3-accuracy-auto': t3_corr_num_auto/incorrect_sol_num,
    }
    return final_stats

def process_eval_results(human_eval_res_path, gpt4_eval_res_path):
    with open(human_eval_res_path) as file:
        human_eval_res = json.load(file)
    task1_true_positive, task1_true_negative = 0, 0
    task2_accy, task3_accy_human = 0, 0
    task3_accy_auto = 0
    step_mapper = {f"step {i}": f"{i}"  for i in range(30)}
    for data in human_eval_res:
        eval_key = [key for key in data.keys() if '_eval_output' in key][0]
        if data[eval_key]['correctness_pred'].lower() == 'incorrect':
            correctness_pred = 'wrong'
        else:
            correctness_pred = data[eval_key]['correctness_pred'].lower()
        if data['model_output_solution_correctness'].lower() == correctness_pred:
            if data['model_output_solution_correctness'].lower() == 'correct':
                task1_true_positive +=1
            else:
                task1_true_negative +=1
                # only if the solution is incorrect and the model agrees on the incorrectness do 
                # we look into task2 and task3 performance
                if data[eval_key]['error_step_pred'].isdigit():
                    error_step_pred = data[eval_key]['error_step_pred'].strip()
                elif data[eval_key]['error_step_pred'].strip().lower() in step_mapper:
                    error_step_pred = step_mapper[data[eval_key]['error_step_pred'].strip().lower()]
                else:
                    error_step_pred = ''
                if error_step_pred == str(data['model_output_solution_first_error_step']):
                    task2_accy += 1
                    if data['error_reason_correctness'].lower() == 'correct':
                        task3_accy_human +=1

    with open(gpt4_eval_res_path) as file:
        gpt4_eval_res = json.load(file)
        for data in gpt4_eval_res:
            if 'correct' in data['gpt4_error_reason_correctness_analysis']['Final Decision'].lower():
                task3_accy_auto += 1
    return {
        't1-tp': task1_true_positive,
        't1-tn': task1_true_negative,
        't2_corr_num': task2_accy,
        't3_corr_num_human': task3_accy_human,
        't3_corr_num_auto': task3_accy_auto
    }
                

def main():
    # TODO: modify this path to your local path 
    mr_gsm8k_path = 'XXX'
    eval_stats = {}
    for entry in os.scandir(f'{mr_gsm8k_path}/eval_results/'):
        if entry.is_file() and entry.name.endswith('eval_results.json'):
            model_name = entry.name.split('_eval_results.json')[0]
            gpt4_eval_res_path = f'{mr_gsm8k_path}/eval_results/gpt4-grading/{entry.name}'
            eval_stats[model_name] = process_eval_results(entry.path, gpt4_eval_res_path)
    
    for model in eval_stats:
        print(f"{model}: {mr_score(model_stat=construct_eval_stats(eval_stats[model]))} ")


if __name__ == '__main__':
    main()