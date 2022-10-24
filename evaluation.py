import os
import pandas as pd
import json
from evaluationTool import *

def precision_recall():
    print("calculate precision recall")
    book_user_test = pd.read_csv(os.path.join("data", "book", "cate3_test.csv"))
    print("test book user Ok")

    recommend_list = {}

    with open(os.path.join("result", "CF", "recommend_list.json"),"r") as jf:
        recommend_list = json.load(jf)
    eva = Evaluator(
        recommend_list=recommend_list, testing_ans=book_user_test
    )
    cmp_diff_n = eva.different_topN(max_topN=1000, rule_table=None)
    with open(os.path.join("result", "CF","metrics","metrics.json"), "w+") as log:
        json.dump(cmp_diff_n, log, indent=4, ensure_ascii=False)

def vis_topn():
    print("vis metrics")
    cmp_diff_n = {}
    with open(os.path.join("result", "CF","metrics","metrics.json"), "r") as log:
        cmp_diff_n = json.load(log)
    
    VisualizationKit.plot_cmp_different_n(
        prec=cmp_diff_n['precision'],
        recall=cmp_diff_n['recall'],
        f1=cmp_diff_n['f1']['val'],
        f1maxpos=cmp_diff_n['f1']['maxpos'],
        max_topN=1000,
        savepath=os.path.join("result","CF", "metrics","metrics.jpg"),
        showinline=False    
    )
    print("PrecisionRecallF1 ", end="")
    print(os.path.join("result","CF", "metrics","metrics.jpg"))
    VisualizationKit.PR_curve(
        precision=cmp_diff_n['precision'], recall=cmp_diff_n['recall'],
        savepath=os.path.join("result","CF","metrics","PR.jpg")
    )
    print("PR ", end="")
    print(os.path.join("result","CF","metrics","PR.jpg"))

    VisualizationKit.ROC(
        cmp_diff_n['fpr'], cmp_diff_n['recall'],
        savepath=os.path.join("result","CF", "metrics","ROC.jpg")
    )
    print("ROC ", end="")
    print(os.path.join("result","CF", "metrics","ROC.jpg"))

if __name__ == "__main__":
    
    if not os.path.exists(os.path.join("result", "CF","metrics","metrics.json")):
        precision_recall()    
    
    vis_topn()