import numpy as np
import torch
import os 
import sys
import pandas as pd
from sklearn.cluster import KMeans


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

def clusterting(k:int, savingpath:os.PathLike):
    pass

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