import json
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(__file__))
import plotutils as VisualizationKit
import numpy as np
import recmetrics


class _Evaluator():
    
    def __init__(self, recommend_list:dict, testing_ans:dict, all_items:list) -> None:
        self.__recommend_list = recommend_list
        self.__gth = testing_ans
        self.__all_items = all_items

    def recommend_topN(self, topN)->dict:
        predi = {}
        for k, v in (self.__recommend_list.items()):
            predi[k] = list(map(lambda x:str(x) , v[:topN]))
        return predi
        
    def _fpr(self, prediction, gth)->float:
        each_fallout = []
        for prei, gthi in zip(prediction,gth):
            fp_count = len(list(b for b in prei if b not in gthi))
            neg = len(list(b for b in self.__all_items if b not in gthi ))
            each_fallout.append(fp_count/neg)
        return np.mean(each_fallout)
    

    def top_N(self, topN)->tuple:   
        
        prediction ,gth =[], []

        for k, v in (self.recommend_topN(topN).items()):

            predi = list(map(lambda x:str(x), v))
            actual = list(map(lambda x:str(x), self.__gth[k]))
    
            prediction.append(predi)
            gth.append(actual)
        
        fpr= self._fpr(prediction, gth)
        prec = recmetrics.recommender_precision(prediction,gth)
        recall = recmetrics.recommender_recall(prediction,gth)

        return prec, recall, fpr
        

    def different_topN(self, max_topN)->dict:
        prec, recall, f1, fpr = [], [], [], []
        for n in tqdm(range(1,max_topN+1)):
            prec_n, recall_n, fpr_n = self.top_N(topN=n)
            prec.append(prec_n)
            recall.append(recall_n)
            fpr.append(fpr_n)
            f1.append(2/((1/(recall_n+1e-10))+(1/(prec_n+1e-10))))
        return {
            'precision':prec, 
            'recall':recall, 
            'fpr':fpr,
            'f1':f1
        }

def precision_recall(predictionfile,gthfile,topN_range,all_items,savepath)->dict:
    
    def read_jsonfile(jsfile:dict)->dict:
        if isinstance(jsfile, dict):
                return jsfile
        else:
            with open(jsfile,"r", encoding='utf-8') as jf:
                return json.load(jf)
    
    print("calculate precision, recall, f1, falsepositive rate")
    groundtruth = read_jsonfile(gthfile)
    recommend_list = read_jsonfile(predictionfile)
    
    eva = _Evaluator(
        recommend_list=recommend_list, 
        testing_ans=groundtruth,
        all_items=all_items
    )
    cmp_diff_n = eva.different_topN(max_topN=topN_range)
    if savepath is not None:
        with open(savepath, "w+") as log:
            json.dump(cmp_diff_n, log, indent=4, ensure_ascii=False)
    
    return cmp_diff_n

def Evaluate(result_root, recommendlist, gth, item_list, topN_range=1000, showinline=False):
    
    def plotmetrics(cmp_diff_n:dict, savepath, showinline=False)->None:
        print("Precision_Recall_F1 :", end=" ")
        VisualizationKit.plot_PRF1_different_n(
            prec=cmp_diff_n['precision'],recall=cmp_diff_n['recall'],
            f1=cmp_diff_n['f1'],
            savepath=savepath,
            showinline=showinline   
        )
        print(savepath)

    def plotPR_curve(cmp_diff_n:dict, savepath, showinline=False):
        print("PR :", end=" ")
        VisualizationKit.PR_curve(
            precision=cmp_diff_n['precision'], recall=cmp_diff_n['recall'],
            savepath=savepath, showinline=showinline
        )
        print(savepath)

    def plotROC( cmp_diff_n:dict, savepath, showinline=False):
        print("ROC :", end=" ")
        VisualizationKit.ROC(
            cmp_diff_n['fpr'], cmp_diff_n['recall'],
            savepath=savepath, showinline=showinline
        )
        print(savepath)
    

    cmp_diff_n = {}
    precal = os.path.join(result_root,"metrics" ,"metrics.json")
   
    print(os.path.join(result_root,"metrics" ,"metrics.json"))
    if not os.path.exists(os.path.join(result_root,"metrics")):
        os.mkdir(os.path.join(result_root,"metrics"))
        
    cmp_diff_n = precision_recall(
        predictionfile=os.path.join(recommendlist),
        gthfile=os.path.join(gth),
        topN_range=topN_range,
        all_items=item_list,
        savepath=os.path.join(result_root, "metrics" ,"metrics.json")
    )    
    
    

    plotmetrics(cmp_diff_n, os.path.join(result_root , "metrics","metrics.jpg"), showinline=showinline)
    plotPR_curve(cmp_diff_n,os.path.join(result_root ,"metrics","PR.jpg"),showinline=showinline)
    plotROC(cmp_diff_n, os.path.join(result_root , "metrics","ROC.jpg"), showinline=showinline)


def item_order(user_rate:np.ndarray, item_id_type:type=str)->list:
    ordered = np.argsort(-user_rate).tolist()
    ordered = list(map(lambda x:item_id_type(x), ordered))
    return ordered