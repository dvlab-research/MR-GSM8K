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

def main():
    
    eval_stats = {
        "Claude-2": construct_eval_stats({
                't1-tp': 962,
                't1-tn': 1056,
                't2_corr_num':331,
                't3_corr_num_human':185,
                't3_corr_num_auto':224,
            }),
        "GPT3-5": construct_eval_stats({
                't1-tp': 1125,
                't1-tn': 621,
                't2_corr_num':179,
                't3_corr_num_human':73,
                't3_corr_num_auto':73,
            }),
        "GPT4":construct_eval_stats({
                't1-tp': 985,
                't1-tn': 1425,
                't2_corr_num':823,
                't3_corr_num_human':677,
                't3_corr_num_auto':732,
            }),
        "WizardMath":construct_eval_stats({
                't1-tp': 1176,
                't1-tn': 43,
                't2_corr_num':6,
                't3_corr_num_human':1,
                't3_corr_num_auto':1,
            }),
        "Mammoth":construct_eval_stats({
                't1-tp': 1410,
                't1-tn': 43,
                't2_corr_num':4,
                't3_corr_num_human':1,
                't3_corr_num_auto':2,
            }),
        "MetaMath":construct_eval_stats({
                't1-tp': 1305,
                't1-tn': 166,
                't2_corr_num':22,
                't3_corr_num_human':6,
                't3_corr_num_auto':7,
            }),
        
        "Lllama2":construct_eval_stats({
                't1-tp': 453,
                't1-tn': 1156,
                't2_corr_num':327,
                't3_corr_num_human':99,
                't3_corr_num_auto':139,
            })
    }
    print('Claude-2', mr_score(model_stat=eval_stats['Claude-2']))
