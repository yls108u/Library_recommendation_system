import os
import json
from EvaTool import *
import sys

def plotmetrics(cmp_diff_n:dict, savepath)->None:
    print("Precision_Recall_F1 :", end=" ")
    VisualizationKit.plot_cmp_different_n(
        prec=cmp_diff_n['precision'],recall=cmp_diff_n['recall'],
        f1=cmp_diff_n['f1'],
        savepath=savepath,
        showinline=False    
    )
    print(savepath)

def plotPR(cmp_diff_n:dict, savepath):
    print("PR :", end=" ")
    VisualizationKit.PR_curve(
        precision=cmp_diff_n['precision'], recall=cmp_diff_n['recall'],
        savepath=savepath, showinline=False
    )
    
    print(savepath)

def plotROC( cmp_diff_n:dict, savepath):
    print("ROC :", end=" ")
    VisualizationKit.ROC(
        cmp_diff_n['fpr'], cmp_diff_n['recall'],
        savepath=savepath, showinline=False
    )
    print(savepath)

def samecourse(result_root):
    cmp_diff_n = {}
    precal = os.path.join(result_root,"metrics" ,"metrics.json")
    if not os.path.exists(precal):
        print(os.path.join(result_root ,"metrics.json"))
        cmp_diff_n = precision_recall(
            predictionfile=os.path.join(result_root , "recommend_list.json"),
            gthcsv=os.path.join("data", "book", "cate3_test.csv"),
            topN_range=1000,
            savepath=os.path.join(result_root ,"metrics.json")
        )    
    else:
        with open(precal, "r") as log:
            cmp_diff_n = json.load(log)
    
    plotmetrics(cmp_diff_n, os.path.join(result_root , "metrics","metrics.jpg"))
    plotPR(cmp_diff_n,os.path.join(result_root ,"metrics","PR.jpg"))
    plotROC(cmp_diff_n, os.path.join(result_root , "metrics","ROC.jpg"))

def simuser(result_root):
    precal = os.path.join(result_root, "metrics", "metrics.json")
    if not os.path.exists(precal):
        cmp_diff_n = {}
        print("cal metrics")
        recommend_list = {}
        with open(os.path.join(result_root, "recommend_list.json"),"r") as jf:
            recommend_list=json.load(jf)
        print("loading recommend list OK")
        k = list(recommend_list.keys())
        v = list(recommend_list.values())
        topK_user_range = list(range(len(v[0])))
        for simsize in topK_user_range:
            print(simsize)
            simuser_rec = dict(zip(k, list( map(lambda vi:vi[simsize], v ) )))
            cmp_diff_n_size = precision_recall(
                predictionfile=simuser_rec,
                gthcsv=os.path.join("data", "book", "cate3_test.csv"),
                topN_range=1000,
                savepath=None
            )
            cmp_diff_n[simsize] =  cmp_diff_n_size
        with open(os.path.join(result_root ,"metrics.json"), "w+") as log:
            json.dump(cmp_diff_n, ensure_ascii=False, indent=4)
        print("write result at :", end = " ")
        print(os.path.join(result_root ,"metrics.json"))

method_evaluation={
    "simuser":simuser,
    "samecourse":samecourse
}

def main(method):
    result_root=os.path.join("result", "CF", f"{method}")
    method_evaluation[method](result_root)

if __name__ == "__main__":
    #method = sys.argv[1]
    method = "samecourse"
    main(method)
    
