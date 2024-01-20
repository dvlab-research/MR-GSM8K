import math

def MCC_score(tp, tn, fp, fn):
    return (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

def mr_score(model_stat, w1=0.2, w2=0.3, w3=0.5):
    mcc_score = MCC_score(model_stat['t1-tp'], model_stat['t1-tn'], model_stat['t1-fp'], model_stat['t1-fn'])
    mr_score_manual = w1 * max(0, mcc_score) + w2 * model_stat['t2-accuracy'] + w3 * model_stat['t3-accuracy']
    mr_score_auto = w1 * max(0, mcc_score) + w2 * model_stat['t2-accuracy'] + w3 * model_stat['t3-accuracy-auto']
    return mr_score_manual, mr_score_auto

def main():
    eval_stats = {
        "Claude-2":{
            't1-tp': 962,
            't1-tn': 1056,
            't1-fp': (1573-1056),
            't1-fn': (1427-962),
            't1-recall': 962/1427,
            't1-precision': 962/(962+1573-1056),
            't2-accuracy': 331/1573,
            't3-accuracy': 185/1573,
            't3-accuracy-auto': 224/1573,
                        
        },
        "GPT3-5":{
            't1-tp': 1125,
            't1-tn': 621,
            't1-fp': (1573-621),
            't1-fn': (1427-1125),
            't1-recall': 1125/1427,
            't1-precision': 1125/(1125+1573-621),
            't2-accuracy': 179/1573,
            't3-accuracy': 73/1573,
            't3-accuracy-auto': 73/1573,           
        },
        "GPT4":{
            't1-tp': 985,
            't1-tn': 1425,
            't1-fp': (1573-1425),
            't1-fn': (1427-985),
            't1-recall': 985/1427,
            't1-precision': 985/(985+1573-1425),
            't2-accuracy': 823/1573,
            't3-accuracy': 677/1573,
            't3-accuracy-auto': 732/1573,            
        },
        "WizardMath":{
            't1-tp': 1176,
            't1-tn': 43,
            't1-fp': (1573-43),
            't1-fn': (1427-1176),
            't1-recall': 1176/1427,
            't1-precision': 1176/(1176+1573-43),
            't2-accuracy': 6/1573,
            't3-accuracy': 1/1573,
            't3-accuracy-auto': 1/1573,            
        },
        "Mammoth":{
            't1-tp': 1410,
            't1-tn': 43,
            't1-fp': (1573-43),
            't1-fn': (1427-1410),
            't1-recall': 1410/1427,
            't1-precision': 1410/(1410+1573-43),
            't2-accuracy': 4/1573,
            't3-accuracy': 1/1573,
            't3-accuracy-auto': 2/1573,            
        },
        "MetaMath":{
            't1-tp': 1305,
            't1-tn': 166,
            't1-fp': (1573-166),
            't1-fn': (1427-1305),
            't1-recall': 1305/1427,
            't1-precision': 1305/(1305+1573-166),
            't2-accuracy': 22/1573,
            't3-accuracy': 6/1573,
            't3-accuracy-auto': 7/1573,         
        },
        "Lllama2":{
            't1-tp': 453,
            't1-tn': 1156,
            't1-fp': (1573-1156),
            't1-fn': (1427-453),
            't1-recall': 453/1427,
            't1-precision': 453/(453+1573-1156),
            't2-accuracy': 327/1573,
            't3-accuracy': 99/1573,
            't3-accuracy-auto': 139/1573,           
        }
    }
    print('Claude-2', mr_score(model_stat=eval_stats['Claude-2']))
