import os
import json
from EvaTool import *

def plotmetrics(cmp_diff_n:dict, savepath)->None:
    print("Precision_Recall_F1 ", end="")
    VisualizationKit.plot_cmp_different_n(
        prec=cmp_diff_n['precision'],recall=cmp_diff_n['recall'],
        f1=cmp_diff_n['f1']['val'],f1maxpos=cmp_diff_n['f1']['maxpos'],
        max_topN=1000,
        savepath=savepath,
        showinline=False    
    )
    print(savepath)

def plotPR(cmp_diff_n:dict, savepath):
    print("PR ", end="")
    VisualizationKit.PR_curve(
        precision=cmp_diff_n['precision'], recall=cmp_diff_n['recall'],
        savepath=savepath, showinline=False
    )
    
    print(savepath)


def plotROC( cmp_diff_n:dict, savepath):
    print("ROC ", end="")
    VisualizationKit.ROC(
        cmp_diff_n['fpr'], cmp_diff_n['recall'],
        savepath=savepath, showinline=False
    )
    print(savepath)


def main():
    cmp_diff_n = {}
    if not os.path.exists(os.path.join("result", "CF","metrics","metrics.json")):
        cmp_diff_n = precision_recall(
            predictionfile=os.path.join("result", "CF", "recommend_list.json"),
            gthcsv=os.path.join("data", "book", "cate3_test.csv"),
            topN_range=1000,
            savepath=os.path.join("result", "CF","metrics","metrics.json")
        )    
    else:
        with open(os.path.join("result", "CF","metrics","metrics.json"), "r") as log:
            cmp_diff_n = json.load(log)
    
    #plotmetrics(cmp_diff_n, os.path.join("result","CF", "metrics","metrics.jpg"))
    #plotPR(cmp_diff_n,os.path.join("result","CF","metrics","PR.jpg"))
    plotROC(cmp_diff_n, os.path.join("result","CF", "metrics","ROC.jpg"))

if __name__ == "__main__":
    main()