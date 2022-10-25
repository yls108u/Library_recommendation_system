from curses import noecho
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import recmetrics
import json
import pandas as pd



class _Evaluator():
    def __init__(self, recommend_list:dict, testing_ans:pd.DataFrame) -> None:
        self.__recommend_list = recommend_list
        self.__gth = testing_ans

    def recommend_topN(self, topN)->dict:
        predi = {}
        for k, v in (self.__recommend_list.items()):
            predi[k] = list(map(lambda x:str(x) , v[:topN]))
        return predi
        
    def _fpr(self, prediction, gth)->float:
        total_books=list(map(lambda x:str(x) , list(i for i in range(self.__gth.shape[0])) ))
        each_fallout = []
        for prei, gthi in zip(prediction,gth):
            fp_count = len(list(b for b in prei if b not in gthi))
            neg = len(list(b for b in total_books if b not in gthi ))
            each_fallout.append(fp_count/neg)
        return np.mean(each_fallout)
    

    def top_N(self, topN, rule_table=None)->tuple:

        def nor(rl, rule_table):
            ret = set()
            for r in rl:
                ret.add(r)
                ass = np.nonzero(rule_table[r])[0].tolist()
                for a in ass:
                    ret.add(a)
            return list(ret)

        
        prediction ,gth =[], []

        for k, v in (self.recommend_topN(topN).items()):
            predi = v
        
            if rule_table is not None:
                predi = nor(predi, rule_table)

            predi = list(map(lambda x:str(x), predi))
            gthi = self.__gth[k].values
            actual = np.where(gthi>0)[0].tolist()
            actual = list(map(lambda x:str(x), actual))
    
            prediction.append(predi)
            gth.append(actual)
        
        fpr= self._fpr(prediction, gth)
        prec = recmetrics.recommender_precision(prediction,gth)
        recall = recmetrics.recommender_recall(prediction,gth)

        return prec, recall, fpr
        

    def different_topN(self, max_topN, rule_table)->dict:
        prec, recall, f1, fpr = [1], [0], [0], [0]
        for n in tqdm(range(1,max_topN+1)):
            prec_n, recall_n, fpr_n = self.top_N(
                topN=n,rule_table=rule_table
            )

            prec.append(prec_n)
            recall.append(recall_n)
            fpr.append(fpr_n)
            f1.append(2/((1/recall_n)+(1/prec_n)))
        return {
            'precision':prec, 
            'recall':recall, 
            'fpr':fpr,
            'f1':f1
        }

def precision_recall(predictionfile,gthcsv,topN_range,savepath)->dict:
    print("calculate precision, recall, f1, falsepositive rate")
    book_user_test = pd.read_csv(gthcsv)
    recommend_list = {}
    if isinstance(predictionfile, dict):
        recommend_list=predictionfile
    else:
        with open(predictionfile,"r") as jf:
            recommend_list = json.load(jf)
    
    eva = _Evaluator(
        recommend_list=recommend_list, 
        testing_ans=book_user_test
    )
    cmp_diff_n = eva.different_topN(max_topN=topN_range, rule_table=None)
    if savepath is not None:
        with open(savepath, "w+") as log:
            json.dump(cmp_diff_n, log, indent=4, ensure_ascii=False)
    
    return cmp_diff_n


class VisualizationKit():
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_cmp_different_n(
        prec, recall, f1, savepath, showinline=True
    ):
        max_topN = len(prec)
        f1maxpos = int(np.argmax(np.array(f1)))
        # f1_max = f1maxpos-1 because starting from N = 0
        plt.figure(dpi=400)
        plt.plot(
            list(k for k in range(0,max_topN)), 
            prec, 
            label = "precision",color="orange"
        )
        plt.plot(
            list(k for k in range(0,max_topN)),
            recall,
            label = "recall",color = "blue"
        )
        plt.plot(
            list(k for k in range(0,max_topN)),
            f1,
            label = "f1",color="green"
        )
        plt.scatter(f1maxpos, f1[f1maxpos], color="purple")
        plt.annotate(
            text=f"Top N : {f1maxpos-1}\nF1 max : {f1[f1maxpos]:.2f}\nPrecision : {prec[f1maxpos]:.2f}\nRecall : {recall[f1maxpos]:.2f}", 
            xytext=(f1maxpos+len(f1)*0.3, f1[f1maxpos]+0.05),color="black",
            xy=(f1maxpos, f1[f1maxpos]),
            arrowprops={
                'width':0.01,'headlength':10,'headwidth':4,
                'facecolor':"black","shrink":0
            },
            va = "center",
            bbox = dict(boxstyle="square", fc="white"),
            fontsize=12
        )
        

        plt.xlabel("N")
        plt.title("Top N")
        plt.legend()
        plt.tight_layout()
        plt.savefig(savepath)
        if not showinline:
            plt.close()
    
    @staticmethod
    def ROC(fpr, recall,savepath, showinline=True):
        parti = np.array(list(i for i in recall))*(1/len(fpr))
        auc = np.sum(parti)
        plt.figure(dpi=400)
        plt.plot(fpr,recall,label=f"AUC: {auc}")
        plt.plot(fpr, fpr, linestyle="--",color="orange")
        #plt.fill_between(fpr, fpr,recall, color="orange",alpha= 0.3)
        plt.ylabel("Recall")
        plt.xlabel("False Positive Rate")
        plt.text(0.8,0, f"AUC : {auc:.2f}",bbox=dict(facecolor='none', edgecolor='black'))
        plt.title("ROC")
        plt.tight_layout()
        plt.savefig(savepath)
        if not showinline:
            plt.close()

    @staticmethod
    def PR_curve(recall, precision, savepath, showinline=True):
        plt.figure(dpi=400)
        plt.plot(recall,precision)
        plt.ylabel("precision")
        plt.xlabel("recall")
        plt.title("PR_curve")
        plt.tight_layout()
        plt.savefig(savepath)
        if not showinline:
            plt.close()
