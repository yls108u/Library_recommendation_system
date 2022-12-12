import numpy as np
import torch
import os 
import sys
import pandas as pd
from sklearn.cluster import KMeans
sys.path.append(os.path.dirname(__file__))
from plotutils import plot_tsne_2d
from dictutils import *

"""
Some functions to process our dataset.
"""



def transpose_df(df:pd.DataFrame, col_to:str)->pd.DataFrame:
    col = df.columns.tolist()
    coldf = pd.DataFrame(data = col, columns=[col_to]) 
    tdf = pd.DataFrame(
        data = df.values.T,
        columns=list(i for i in range(df.shape[0]))
    )
    return pd.concat((coldf, tdf), axis = 1)

def clusterting(items:list, x:np.ndarray, k:int, savingpath:os.PathLike, visshow=False):
    cluster = KMeans(n_clusters=k)
    print("clutsering ..")
    cluster.fit(x)
    y = cluster.predict(x)
    print("OK ..")

    cluster_result = {}
    for i in range(k):
        cluster_result[i] = []
    y_ = y.tolist()
    for idx, i in enumerate(items):
        cluster_result[y_[idx]].append(i)
    
    if not os.path.exists(savingpath):
        os.mkdir(savingpath)
    writejson(cluster_result, os.path.join(savingpath, "cluster.json"))
    print("visualization ..")

    plot_tsne_2d(
        X=x, name=f"clustering_{k}", labels=y,
        savepath=os.path.join(savingpath,f"clustering_{k}.jpg"),
        showinline=visshow
    )    


class Dataset():

    def __init__(self, datafolder:dict) -> None:
        print("build dataset")
        self._data = self._read_data(datafolder)
        self._dname =list(self._data.keys())
        
    def _read_data(self, datafolder:dict)->dict:
        ret = {}
        for dataname, path in datafolder.items():
            print(f"read {dataname}:{path}")
            ret[dataname] = pd.read_csv(path)
            print("..OK")
        return ret
    
    def dataname(self) -> list:
        return self._dname

    def getdata(self, dataname:str)-> pd.DataFrame:
        if dataname == "all_data":
            return self._data.copy()
        return self._data[dataname].copy()
    
    def extraction_value(self,dname:str, dropout_col:list, to_torch=False):
        
        if dname not in self._dname:
            print("no such a data")
            raise KeyError
      
        value =(
            self._data[dname].drop(columns=dropout_col)
        ).values
    
        if to_torch:
            return torch.tensor(value, dtype=torch.double)
        return value

    def mask_dataset(self, dname:str)->None:
        """
        To make the target df 's value to 0
        (e.g.: mask up the ground truth of testing data)
        """
        if dname not in self._dname:
            print(f"No such {dname} data")
            raise KeyError
        itsuid = self._data[dname].uid.tolist()
        self._data[dname]=self._data[dname].drop(col=['uid'])
        mask_df = pd.DataFrame(
            data=(self._data[dname]).values*0.0,
            columns=self._data[dname].columns.tolist()
        )
        mask_df['uid'] = itsuid
        col_order =mask_df.columns.tolist()[-1:]+ mask_df.columns.tolist()[:-1]
        mask_df = mask_df[col_order]
        self._data[dname] = mask_df