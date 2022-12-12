import torch 
import numpy as np
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(__file__))
from utils.plotutils import plotLoss


class ALS_MF():
    
    def __init__(self,R:torch.Tensor, latency:int=40, l2_reg:float=0.1,) -> None:
        
        self._R:torch.Tensor = R.to(dtype = torch.double)
    
        self._predicton_R = self._R
        self._predicton_Rt = self._predicton_R.T

        self._latency:int = latency 
        self._l2_reg:float = l2_reg
        self._I = torch.eye(self._latency, dtype=torch.double)
        self._embedding = {
            "user": torch.tensor(np.random.random((R.size()[0], latency)), dtype=torch.double),
            "item": torch.tensor(np.random.random((R.size()[1], latency)), dtype=torch.double)
        }

    def user_latency(self, asnp=False):
        if asnp:
            return self._embedding['user'].cpu().numpy()
        return torch.clone(self._embedding['user'].cpu())
    
    def item_latency(self,asnp=False):
        if asnp:
            return self._embedding['item'].cpu().numpy()
        return torch.clone(self._embedding['item'].cpu())
    
    def prediction(self):
        return ( self._embedding['user']@(self._embedding['item'].T) ).cpu()

    def mse(self):
        return torch.mean((self.prediction()-self._R)**2)
    
    def _step(self, target, constant, rate,device):
        
        A = (self._embedding[constant].T)@self._embedding[constant]+self._l2_reg*self._I
        A = torch.tensor(np.linalg.inv(A.cpu()),dtype=torch.double).to(device=device)
        #print(f"A:{A.dtype}, rate:{rate.dtype}, constant:{constant.dtype}")
    
        self._embedding[target] = rate@self._embedding[constant]@A
        

    def _early_return_flag (self, e, improve, thr = 0.0001)->int:
        if improve < 0:
            return 0
        if improve < thr:
            return e + 1
        return 0

    def train(self, device:torch.device, max_iteration:int=5)->list:
        
        self._predicton_R = self._predicton_R.to(device=device)
        self._predicton_Rt = self._predicton_Rt.to(device=device)
        self._I = self._I.to(device=device)
        
        self._embedding['user'] = self._embedding['user'].to(device=device)
        self._embedding['item'] = self._embedding['item'].to(device=device)
        

        pbar = tqdm(range(max_iteration))
        criteria=self.mse().item()
        print(f"random loss: {criteria}")
        mse_history = [criteria]
        last = criteria
        improve = np.inf
        early = 0

        for _ in pbar:

            self._step(target='user',constant='item',rate=self._predicton_R,device=device,epoch=_)
            self._step(target='item', constant='user',rate=self._predicton_Rt,device=device, epoch=_)
            criteria=self.mse().item()
            mse_history.append(criteria)
            improve  = last - criteria
            if improve > 0:
                last = criteria

            early = self._early_return_flag(e = early,improve = improve)
            pbar.set_postfix(
                ordered_dict={
                    'mse':f"{criteria:.3f}", 
                    "improve" :f"{improve:.4f}",
                    "early":f"{early}"
                }
            )
            if early == 3:
                return mse_history
    
        self._embedding['user'] = self._embedding['user'].cpu()
        self._embedding['item'] = self._embedding['item'].cpu()
        self._I = self._I.cpu()
        self._predicton_R = self._predicton_R.cpu()
        return mse_history


class WeightedALS_MF(ALS_MF):
    
    def __init__(self, R: torch.Tensor,fill_empty, latency: int = 40, l2_reg: float = 0.1,w_obs=1.0, w_m=1.0) -> None:
        
        super().__init__(R, latency, l2_reg)
        
        self._w =torch.ones(R.size(), dtype=torch.double)*w_obs 
        unobs_index = torch.nonzero(self._R == 0.0)
        self._wm = w_m
        self._w[unobs_index[:, 0], unobs_index[:, 1]] = w_m
        self._fill_empty = fill_empty
        self._predicton_R = self._R-self._fill_empty
        self._predicton_Rt = self._predicton_R.T
    
    def prediction(self):
        return super().prediction()+self._fill_empty

    def train(self, device: torch.device, max_iteration: int = 5) -> list:
        
        self._w=self._w.to(device=device)
        h =super().train(device, max_iteration)
        self._w=self._w.cpu()
        return h
    
    def _step(self, target, constant, rate, device, **kwarg):
        row_bar = tqdm(range(self._embedding[target].size()[0]))
        
        w = None
        if target == "user":
            w = self._w
        elif target == "item":
            w = self._w.T
        
        ctc= self._wm*self._embedding[constant].T@self._embedding[constant]
        
        for r in row_bar:

            obs = torch.nonzero(rate[r]>0).T
            wrobs = w[r][obs].T
            constant_obs = self._embedding[constant][obs][0]
            r_obs = rate[r][obs]-self._fill_empty
            
            z = r_obs@((wrobs.repeat(1,self._latency))*constant_obs)
            A = (torch.tensor(np.linalg.inv((
                ctc-self._wm*constant_obs.T@constant_obs+\
                ((constant_obs.T*wrobs.sum(dim=0))@constant_obs)+\
                self._l2_reg*torch.sum(wrobs)*self._I).cpu()
                ),
                dtype=torch.double, device=device)
            )
            self._embedding[target][r]=z@A
            row_bar.set_postfix(
                ordered_dict={'target':target,'epoch':kwarg['epoch']}
            )

        
def test():
    testing_matrix = torch.randint(
        low=0, high=5, size=(1000,5000),
        dtype=torch.double
    )

    walsmf = WeightedALS_MF(
        R=testing_matrix, 
        fill_empty=torch.mean(testing_matrix).item()/2,
        w_m = 0.001
    )

    h = walsmf.train(device=torch.device('cuda:6'))
    
    
    plotLoss(
        loss=h,savename=os.path.join("test", "test_wals_mse.jpg"),
        showinline=False
    )


if __name__ == "__main__":
    test()

