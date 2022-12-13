import numpy as np
import torch
import os 
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
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

    def getdata(self, dataname:str, normalize_value=False)-> pd.DataFrame:

        if dataname not in self._data:
            print(f"No {dataname} such a data")
            raise KeyError
        df = self._data[dataname].copy()
        if normalize_value:
            uid = df.uid.tolist()
            df= df.drop(columns=['uid'])
            v = normalize(df.values, axis=1, norm="l1")
            df_nor = pd.DataFrame(data=v, columns=df.columns)
            df_nor['uid'] = uid
            col_order = df_nor.columns.tolist()[-1:]\
                +df_nor.columns.tolist()[:-1]
            return df_nor[col_order]

        return df
    
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

    def mask_dataset(self, dataname:str)->pd.DataFrame:
        """
        To make the target df 's value to 0
        (e.g.: mask up the ground truth of testing data)
        """
        if dataname not in self._dname:
            print(f"No such {dataname} data")
            raise KeyError
        df = self._data[dataname].copy()
        uid = df.uid.tolist()
        df = df.drop(columns = ['uid'])
        df_mask = pd.DataFrame(
            data=df.values*0.0,
            columns=df.columns.tolist()
        )
        df_mask['uid'] = uid
        col_order = df_mask.columns.tolist()[-1:]+df_mask.columns.tolist()[:-1]
        return df_mask[col_order]
    
def Crossdomain(dataset_:Dataset=None, datafolder:dict=None):
    
    dataset = None
    if dataset_ is not None:
        dataset=dataset_
    else:
        dataset = Dataset(datafolder=datafolder)

    user_book_all = pd.concat(
        [dataset.getdata(dataname="training_user_book", normalize_value=True), 
        dataset.mask_dataset(dataname="testing_user_book")], 
        axis=0
    )
    user_course_all = pd.concat(
        [dataset.getdata(dataname="training_user_course"), 
        dataset.getdata(dataname="testing_user_course")], 
        axis=0
    )
    crossdomain = pd.concat(
        [user_book_all, user_course_all.drop(columns=['uid'])],
        axis=1
    )
    return crossdomain

