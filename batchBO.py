from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.cached_cholesky import CachedCholeskyMCAcquisitionFunction
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.utils import prune_inferior_points
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionFunction


class MyqExpectedImprovement(MCAcquisitionFunction):


    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
     
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)
        
        obj =  (self.best_f.unsqueeze(-1).to(obj)-obj).clamp_min(0) 
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return q_ei
    

class MySlogEI(MCAcquisitionFunction):
 

    def __init__(
        self,
        model: Model,
        C: Union[float, Tensor],
        warp_mean: Union[float, Tensor],
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
     
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))
        self.register_buffer("C", torch.as_tensor(C, dtype=float))
        self.register_buffer("warp_mean", torch.as_tensor(warp_mean, dtype=float))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:

        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)
        obj = (torch.exp(obj+self.warp_mean)) - self.C
        obj = (self.best_f.unsqueeze(-1).to(obj)-obj).clamp_min(0) 
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        
        return q_ei
    
    
class MySlogTEI(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        C: Union[float, Tensor],
        warp_mean: Union[float, Tensor],
        best_f: Union[float, Tensor],
        f_star: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
     
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))
        self.register_buffer("f_star", torch.as_tensor(f_star, dtype=float))
        self.register_buffer("C", torch.as_tensor(C, dtype=float))
        self.register_buffer("warp_mean", torch.as_tensor(warp_mean, dtype=float))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:

        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)

        obj = (torch.exp(obj+self.warp_mean)) - self.C
        obj = (self.best_f.unsqueeze(-1).to(obj)-obj).clamp_min(0) - (self.f_star.unsqueeze(-1).to(obj)-obj).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        
        return q_ei
    
    
import botorch
from known_bound.acquisition_function import EI_acquisition_opt,MES_acquisition_opt,LCB_acquisition_opt,ERM_acquisition_opt,SLogTEI_acquisition_opt,SLogEI_acquisition_opt
from known_bound.utlis import  get_initial_points,transform,opt_model_MLE,opt_model_MAP
import numpy as np
import GPy
import torch
from botorch.test_functions import Shekel,Ackley,Beale,Branin,Rosenbrock,SixHumpCamel,Hartmann,Powell,DixonPrice,Levy,StyblinskiTang,Griewank
import obj_functions.push_problems
import obj_functions.lunar_lander
import obj_functions.rover_problems
from  obj_functions.obj_function import Sphere
from botorch.utils.transforms import unnormalize,normalize
from known_bound.SLogGP import SLogGP
import scipy 
from botorch.models import FixedNoiseGP
from gpytorch.kernels import  RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.kernels.scale_kernel import ScaleKernel

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement
)
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler


import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('lengthscale').disabled = True
logging.getLogger('variance').disabled = True
logging.getLogger('psi').disabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


function_information = []




# temp={}
# temp['name']='Branin2D' 
# temp['function'] = Branin(negate=False)
# temp['fstar'] =  0.397887 
# function_information.append(temp)

# temp={}
# temp['name']='Levy2D' 
# temp['function'] = Levy(dim=2,negate=False)
# temp['fstar'] =  0. 
# function_information.append(temp)

# temp={}
# temp['name']='DixonPrice4D' 
# temp['function'] = DixonPrice(dim=4,negate=False)
# temp['fstar'] =  0. 
# function_information.append(temp)

# temp={}
# temp['name']='Ackley6D'   
# temp['function'] = Ackley(dim=6,negate=False)
# temp['fstar'] =  0. 
# function_information.append(temp)

# temp={}
# temp['name']='Hartmann6D' 
# temp['function'] = Hartmann(dim=6,negate=False)
# temp['fstar'] =  -3.32237
# function_information.append(temp)

temp={}
temp['name']='Shekel4D' 
temp['function'] = Shekel(m=5,negate=False)
temp['fstar'] =  -10.1532
function_information.append(temp)





for information in function_information:

    fun = information['function']
    dim = fun.dim
    bounds = fun.bounds
    standard_bounds=np.array([0.,1.]*dim).reshape(-1,2)
    
    standard_bounds_torch = torch.tensor([0.,1.]*dim).reshape(-1,2).T.to(device)
    
    n_init = 4*dim 

    fstar = information['fstar']
    
    print('fstar is: ',fstar)

    step_size = 10
    if dim == 2: 
        iter_num = 10
    else:
        iter_num = 15
    N = 100
        
    lengthscale_range = [0.001,2]
    variance_range = [0.001**2,20]
    noise = 6e-6
    
    print(information['name'])
        
    
    ############################# qEI ###################################
    qEI_GP = []

    for exp in range(N):
        
        noise = 6e-6
        
        print(exp)
        
        seed = exp

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)

        best_record = [Y_BO.min().item()]
        np.random.seed(1234)
        
        print(best_record[-1])

        for i in range(iter_num):
 
                Y_mean =  Y_BO.mean()
                Y_std = Y_BO.std()

                fstar_rescale = (fstar -Y_mean) / Y_std
                train_Y = (Y_BO -Y_mean) / Y_std
                train_X = normalize(X_BO, bounds)
                
                
                train_Y = train_Y.numpy()  # to make it a maximization problem
                train_X = train_X.numpy()
                
                # train the GP          
                parameters = opt_model_MLE(train_X,train_Y,dim,'GP',noise=noise,seed=i,lengthscale_range=lengthscale_range,variance_range=variance_range)
                    
                lengthscale = parameters[0]
                variance = parameters[1]

            
                train_yvar = torch.tensor(noise, device=device, dtype=dtype)    
                model = FixedNoiseGP(torch.tensor(train_X), torch.tensor(train_Y), train_yvar.expand_as(torch.tensor(train_Y)),
                                          covar_module=ScaleKernel(RBFKernel()),
                                          mean_module=ZeroMean()).to(device)
                
                model.covar_module.base_kernel.lengthscale = lengthscale #torch.tensor(lengthscale)
                model.covar_module.outputscale = variance #torch.tensor(variance)  #I have checked that outputscale is variance
                
                model.eval()
                torch.manual_seed(exp+i)
                
                qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]),seed=0, resample=False)
                
                qEI = MyqExpectedImprovement(
                            model=model,
                            best_f=torch.tensor(train_Y).min(),
                            sampler=qmc_sampler,
    
                        )
            

                
                candidates, _ = optimize_acqf(
                        acq_function=qEI,
                        bounds=standard_bounds_torch,
                        q=step_size,
                        num_restarts=3*dim,
                        raw_samples=50,  
                        options={"batch_limit": 5, "maxiter": 200},
                    )

                X_next = unnormalize(torch.tensor(candidates), bounds).reshape(-1,dim)    
                Y_next = torch.tensor(
                        [fun(x) for x in X_next], dtype=dtype, device=device
                    ).reshape(-1,1)        
         

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                
                print(best_record[-1])
                
                noise = variance*10**(-5)   #adaptive noise
                noise = np.round(noise, -int(np.floor(np.log10(noise))))
     
                
        best_record = np.array(best_record) 
        qEI_GP.append(best_record)

    np.savetxt('result/'+information['name']+'_qEI10', qEI_GP, delimiter=',')
    

  ############################# qSlogTEI ###################################
    qSlogTEI = []


    for exp in range(N):
        
        noise = 6e-6
    
        seed = exp
    
        print(exp)

        X_BO = get_initial_points(bounds, n_init,device,dtype,seed=seed)
    
    
        Y_BO = torch.tensor(
            [fun(x) for x in X_BO], dtype=dtype, device=device
        ).reshape(-1,1)



        best_record = [Y_BO.min().item()]
        print(best_record[-1])
        np.random.seed(1234)
    
        tolerance_level = 2.5

    
        uncertainty = 1
    
        boundarys = []
        variances = []

        Train = False

        for i in range(iter_num):

        
                train_Y = Y_BO.numpy()
           
                Y_min = np.min(train_Y)
                Y_std = np.std(train_Y-Y_min)
                
                fstar_shifted = fstar -Y_min # shifted lower bound
                train_Y = train_Y - Y_min  # shift Y
            
                fstar_shifted = fstar_shifted/Y_std
                train_Y = train_Y/Y_std
                
                
                train_X = normalize(X_BO, bounds)
                train_X = train_X.numpy()
            
                lower = -np.min(train_Y)+10**(-6)
                if Y_std<=2.0:
                    upper = -fstar_shifted+100 
                else:
                    upper = -fstar_shifted+30

                c_range = [lower,upper]
            
                mu_prior = np.log(-fstar_shifted)
                sigma_prior = np.sqrt(2*(np.log(-fstar_shifted+0.1/Y_std)-mu_prior)) * uncertainty
                
                prior_parameter = [mu_prior,sigma_prior]
                     
                
                if sigma_prior<7.5:
                            
                    parameters = opt_model_MAP(train_X,train_Y,dim,lengthscale_range,variance_range,c_range,
                                                    prior_parameter,noise=noise,seed=i)

                    c = parameters[2]
                    MAP = True
                
                    if abs(np.log(c) - mu_prior)>tolerance_level*sigma_prior :
                                                            
                        temp = (abs(np.log(c) - mu_prior))/ sigma_prior 
                        uncertainty *= temp 
                    
                        MAP = False
                        parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                                                    lengthscale_range=lengthscale_range,
                                                    variance_range=variance_range,c_range=c_range)  
                    
                    if MAP:    
                        if parameters[1]<0.25**2:
                                parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                                        lengthscale_range=lengthscale_range,
                                        variance_range=variance_range,c_range=c_range)
                        
                    
                
                else:

                    parameters = opt_model_MLE(train_X,train_Y,dim,'SLogGP',noise=noise,seed=i,
                                                        lengthscale_range=lengthscale_range,
                                                        variance_range=variance_range,c_range=c_range)
                
            
                lengthscale = parameters[0]
                variance = parameters[1]
                c = parameters[2]

                
                warp_Y = np.log(train_Y+c)
                mean_warp_Y = np.mean(warp_Y) # use to predict mean
                warp_Y_standard = warp_Y-mean_warp_Y 
                            
                train_yvar = torch.tensor(noise, device=device, dtype=dtype)    
                model = FixedNoiseGP(torch.tensor(train_X), torch.tensor(warp_Y_standard), 
                                     train_yvar.expand_as(torch.tensor(warp_Y_standard)),
                                         covar_module=ScaleKernel(RBFKernel()),
                                         mean_module=ZeroMean()).to(device)
                
                model.covar_module.base_kernel.lengthscale = lengthscale
                model.covar_module.outputscale = variance  #I have checked that outputscale is variance
                
                model.eval()
                torch.manual_seed(exp+i)
                qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]),seed=0, resample=False)
                                
                if -c>=fstar_shifted:
                    print('logEI')
                    qEI = MySlogEI(        
                        model=model,
                        C = c,
                        warp_mean = mean_warp_Y,
                        best_f=torch.tensor(train_Y).min(),
                        sampler=qmc_sampler)
                else:
                    print('logTEI')
                    qEI = MySlogTEI(   model=model,
                            C = c,
                            warp_mean = mean_warp_Y,
                            best_f=torch.tensor(train_Y).min(),
                            f_star=fstar_shifted,
                            sampler=qmc_sampler)
                
                
                candidates, _ = optimize_acqf(
                        acq_function=qEI,
                        bounds=standard_bounds_torch,
                        q=step_size,
                        num_restarts=3*dim,
                        raw_samples=50,  
                        options={"batch_limit": 5, "maxiter": 200},
                    )
                

                X_next = unnormalize(torch.tensor(candidates), bounds).reshape(-1,dim)    
                Y_next = torch.tensor(
                        [fun(x) for x in X_next], dtype=dtype, device=device
                    ).reshape(-1,1)        
         

                # Append data
                X_BO = torch.cat((X_BO, X_next), dim=0)
                Y_BO = torch.cat((Y_BO, Y_next), dim=0)
                
                best_record.append(Y_BO.min().item())
                
                print(best_record[-1])
                
                noise = variance*10**(-5)   #adaptive noise
                noise = np.round(noise, -int(np.floor(np.log10(noise))))
 

        best_record = np.array(best_record)    
        qSlogTEI.append(best_record)
        
    np.savetxt('result/'+information['name']+'_qSlogTEI10', qSlogTEI, delimiter=',')
        