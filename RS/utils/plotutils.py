import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne_2d(X,name,savepath, labels=None, tsneresult=False, showinline=True):
    xtsne=TSNE(n_components=2).fit_transform(X)
    plt.figure(dpi=800)
    if labels is None:
        plt.scatter(xtsne[:,0], xtsne[:,1])
    else:
        plt.scatter(
            xtsne[:,0], xtsne[:,1], s=10, 
            c = labels, cmap ='viridis'
        )
    
    plt.title(name)
    plt.legend()
    plt.savefig(savepath)
    if not showinline:
        plt.close()
    if tsneresult:
        return xtsne

def plot_PRF1_different_n(prec, recall, f1, savepath, showinline=True, annotate=True):
        
    max_topN = len(prec)
    f1maxpos = int(np.argmax(np.array(f1)))
    # f1_max = f1maxpos-1 because starting from N = 0
    x_axis = list(k for k in range(1,max_topN+1))
    plt.figure(dpi=800)
    plt.plot(x_axis, prec, label = "precision",color="orange")
    plt.plot(x_axis,recall,label = "recall",color = "blue")
    plt.plot(x_axis,f1,label = "f1",color="green")
    plt.scatter(f1maxpos, f1[f1maxpos], color="purple")
    if annotate:
        plt.annotate(
        text=f"Top N : {f1maxpos-1}\nF1 max : {f1[f1maxpos]:.2f}\nPrecision : {prec[f1maxpos]:.2f}\nRecall : {recall[f1maxpos]:.2f}", 
        xytext=(f1maxpos+len(f1)*0.3, f1[f1maxpos]+0.1),color="black",
        xy=(f1maxpos, f1[f1maxpos]),
        arrowprops={
            'width':0.01,'headlength':10,'headwidth':4,
            'facecolor':"black","shrink":0
        },
        va = "center",
        bbox = dict(boxstyle="square", fc="white"),fontsize=9
    )
        
    plt.xlabel("N")
    plt.title(f"Top N: ~ {max_topN}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    if not showinline:
        plt.close()

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
   
def plotLoss(loss:list, savename:os.PathLike,showinline=True, require=min)->None:
    
    epoch = list(e for e in range(len(loss)))
    best_v = require(loss)
    plt.figure(dpi=800)
    plt.plot(epoch, loss, label = f"best: {best_v}")
    plt.legend()
    plt.savefig(savename)
    if not showinline:
        plt.close()

def plot_comparison(x:list, labels:list, title:str, savename:os.PathLike, showinline=True):
    e = list(_ for _ in range(1,len(x[0])+1))
    plt.figure(dpi=800)
    for xi, label in zip(x, labels):
        plt.plot(e, xi, label = label)
    plt.title(title)
    plt.legend()
    plt.savefig(savename)
    if not showinline:
        plt.close()