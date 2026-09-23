from eval import ruler, longeval, longbench, locomo
import prepare_babilong as babi

def scores(row,text):
    b=row['benchmark']
    if b=='ruler': return dict(score=ruler._string_match_all_one(text,row['answers']))
    if b=='longeval': return dict(score=float(longeval.extract_prediction(text)==row['answers'][0]))
    if b=='longbench': return dict(score=longbench.compute_f1_multi(text,row['answers']))
    if b=='babilong': return dict(score=babi.score_prediction(text,row['extra'],row['task']))
    sc=locomo.score_sample(dict(pred=text,answers=row['answers'],**row['extra']))
    # Lexical F1 is a diagnostic, never a substitute for the Table1 Judge column.
    return dict(score=None, lexical=sc, judge_score=sc['acc'] if row['extra']['is_abstention'] else None,
                judge_status='local_abstention' if row['extra']['is_abstention'] else 'pending')
