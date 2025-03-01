import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from numpy import genfromtxt
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier # Feature selection
from scipy.stats import zscore
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes



class XGBoost:
    def __init__(self,task,seed=1):
        # define the search range for each variable
        self.bounds = torch.tensor(np.asarray([
                                [0.,10.],  # alpha
                                  [0.,10.],# gamma 
                                  [5.,15.], #max_depth
                                  [1.,20.],  #min_child_weight
                                  [0.5,1.],  #subsample
                                  [0.1,1] #colsample
                                 ]).T)
            
        self.dim = 6
        self.fstar = 100

        self.seed= seed
        
        self.task = task
        
        DATA_LOADERS = {
                "digits": (datasets.load_digits()),
                "iris": (datasets.load_iris()),
                "wine": (datasets.load_wine()),
                "breast": (datasets.load_breast_cancer()),
            }
                
        
        if task == 'skin':
            data = np.genfromtxt('obj_functions/Skin_NonSkin.txt', dtype=np.int32)
            outputs = data[:,3]
            inputs = data[:,0:3]
            X_train1, _, y_train1, _ = train_test_split(inputs, outputs, test_size=0.85, random_state=self.seed)
            y_train1 = y_train1-1
            
            self.X_train1 = X_train1
            self.y_train1 = y_train1
            

                        
        elif task == 'bank':
            data = pd.read_csv('obj_functions/BankNote_Authentication.csv')
            X = data.loc[:, data.columns!='class']
            y = data['class']
            self.y_train1 = np.array(y)
            self.input = np.array(X)
            
            #preprocessing
            scaler = preprocessing.StandardScaler().fit(self.input)
            self.X_train1 = scaler.transform(self.input)
            

 
        elif task == 'breast' or task == 'iris' or task == 'wine' or task == 'digits':
            data_loader = DATA_LOADERS[task]
            self.input = data_loader.data
            self.y_train1 = data_loader.target

            #preprocessing
            scaler = preprocessing.StandardScaler().fit(self.input)
            self.X_train1 = scaler.transform(self.input)
            
    def __call__(self, X):
        
        X = X.numpy().reshape(6,)

        
        alpha,gamma,max_depth,min_child_weight,subsample,colsample=X[0],X[1],X[2],X[3],X[4],X[5]
        
        if self.task == 'bank'  or self.task == 'skin' or self.task == 'breast' or self.task == 'diabetes' or self.task == 'wine':
            reg = XGBClassifier(reg_alpha=alpha, gamma=gamma, max_depth=int(max_depth), subsample=subsample, 
                        min_child_weight=min_child_weight,colsample_bytree=colsample, n_estimators = 2, random_state=self.seed, objective = 'binary:logistic', booster='gbtree',eval_metric='logloss',silent=None)
            score = np.array(cross_val_score(reg, X=self.X_train1, y=self.y_train1).mean())
        elif self.task == 'iris' or 'digits':
            reg = XGBClassifier(reg_alpha=alpha, gamma=gamma, max_depth=int(max_depth), subsample=subsample, 
                    min_child_weight=min_child_weight,colsample_bytree=colsample, n_estimators = 2, random_state=self.seed, objective = 'multi:softmax', booster='gbtree',eval_metric='logloss',silent=None)
            score = np.array(cross_val_score(reg, X=self.X_train1, y=self.y_train1).mean())
             
        return 100-torch.tensor([score*100])
    
    

from botorch.test_functions.synthetic import SyntheticTestFunction

# from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
from botorch.exceptions.errors import InputDataError
from botorch.test_functions.base import BaseTestProblem, ConstrainedBaseTestProblem
# from botorch.test_functions.utils import round_nearest
from torch import Tensor


class Sphere(SyntheticTestFunction):


    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        if bounds is None:
            bounds = [(-10., 10.) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)



    def evaluate_true(self, X: Tensor) -> Tensor:
        res = torch.linalg.norm(X+3., dim=-1)
        return res**2
    
    

# Classification: LogisticRegression, SVM, DT
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import torch
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

class LogisticRegression_Classification:
    def __init__(self,dataset_name,penalty):
        
        self.penalty = penalty
        
        DATA_LOADERS = {
                        "digits": (datasets.load_digits()),
                        "iris": (datasets.load_iris()),
                        "wine": (datasets.load_wine()),
                        "breast": (datasets.load_breast_cancer()),
                        "fetch_olivetti_faces": (datasets.fetch_olivetti_faces()),
                    }
        
        data_loader = DATA_LOADERS[dataset_name]
        self.input = data_loader.data
        self.target = data_loader.target

        #preprocessing
        scaler = preprocessing.StandardScaler().fit(self.input)
        self.input_scaled = scaler.transform(self.input)

        self.CV_SPLITS = 5

        # define the search range for each variable
        self.bounds = torch.tensor(np.asarray([
                                [-2.,2.],  # C
                                  [-2.,2.],# intercept_scaling 
                                 ]).T)  
        self.dim = 2
        self.fstar = 0        
      
            
    def __call__(self, X):
        
        X = X.numpy().reshape(self.dim,)

        C,intercept_scaling=10**(X[0]),10**(X[1])
        
        model = LogisticRegression(penalty=self.penalty,C=C,intercept_scaling=intercept_scaling )

        np.random.seed(1234)
        
        res = cross_val_score(model, self.input_scaled, self.target, cv=self.CV_SPLITS)
        average_res = np.mean(res)
        

      
        return 100-torch.tensor([average_res*100])
    
    

class SVM_Classification:
    def __init__(self,task):
        
        self.task = task
        
        DATA_LOADERS = {
                        "digits": (datasets.load_digits()),
                        "iris": (datasets.load_iris()),
                        "wine": (datasets.load_wine()),
                        "breast": (datasets.load_breast_cancer()),
                    }
        
        data_loader = DATA_LOADERS[task]
        self.input = data_loader.data
        self.target = data_loader.target

        #preprocessing
        scaler = preprocessing.StandardScaler().fit(self.input)
        self.input_scaled = scaler.transform(self.input)

        self.CV_SPLITS = 5
        

        # define the search range for each variable
        self.bounds = torch.tensor(np.asarray([
                                [0.,3.],  # C
                                  [-4.,-3.],# gamma 
                                 [-5.,-1.], #tol
                                 ]).T)  
        self.dim = 3
        self.fstar = 0        
      
            
    def __call__(self, X):
        
        X = X.numpy().reshape(self.dim,)

        C,gamma,tol=10**(X[0]),10**(X[1]),10**(X[2])
        
        model = SVC(C=C,gamma=gamma,tol=tol)

        np.random.seed(1234)
        
        res = cross_val_score(model, self.input_scaled, self.target, cv=self.CV_SPLITS)
        average_res = np.mean(res)
        

      
        return 100-torch.tensor([average_res*100])
    
    
class DT_Classification:
    def __init__(self,dataset_name):
        

        
        DATA_LOADERS = {
                        "digits": (datasets.load_digits()),
                        "iris": (datasets.load_iris()),
                        "wine": (datasets.load_wine()),
                        "breast": (datasets.load_breast_cancer()),
                    }
        
        data_loader = DATA_LOADERS[dataset_name]
        self.input = data_loader.data
        self.target = data_loader.target

        #preprocessing
        scaler = preprocessing.StandardScaler().fit(self.input)
        self.input_scaled = scaler.transform(self.input)

        self.CV_SPLITS = 5
        



        # define the search range for each variable
        self.bounds = torch.tensor(np.asarray([
                                [1.,15.],  # max_depth
                                  [0.01,0.99],# min_samples_split 
                                 [0.01,0.49], #min_samples_leaf
                                  [0.01,0.49],#min_weight_fraction_leaf
                                  [0.01,0.99], #max_features
                                  [0.0, 0.5], #min_impurity_decrease
                                 ]).T)  
        self.dim = 6
        self.fstar = 0        
      
            
    def __call__(self, X):
        
        X = X.numpy().reshape(self.dim,)

        max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_features,min_impurity_decrease=int(X[0]),X[1],X[2],X[3],X[4],X[5]
        
        model = DecisionTreeClassifier (max_depth=max_depth,min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,
                                       max_features=max_features,min_impurity_decrease=min_impurity_decrease)

        np.random.seed(1234)
        
        res = cross_val_score(model, self.input_scaled, self.target, cv=self.CV_SPLITS)
        average_res = np.mean(res)
        

      
        return 100-torch.tensor([average_res*100])



# from typing import Optional
import psutil
# import torch
# from botorch.test_functions.base import BaseTestProblem
# from torch import Tensor
# import gc
# from pde import trackers
# try:
#     from pde import PDE, FieldCollection, ScalarField, UnitGrid, MemoryStorage
# except:
#     pass

# class PDEVar(BaseTestProblem):
#     def __init__(
#         self,
#         noise_std: Optional[float] = None,
#         negate: bool = False,
#         aggregate: bool = False
#     ) -> None:

#         self.dim = 4
#         self._bounds = [
#             [0.1, 5.0], 
#             [0.1, 5.0], 
#             [0.01, 5.0], 
#             [0.01, 5.0],    
#         ]
#         self.num_objectives = 1
#         super().__init__(
#                 negate=negate, noise_std=noise_std)

#     def Simulator(self, tensor):
#         torch.manual_seed(1234)
#         a = tensor[0].item()
#         b = tensor[1].item()
#         d0 = tensor[2].item()
#         d1 = tensor[3].item()




#         eq = PDE(
#             {
#                 "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
#                 "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
#             }
#         )

#         # initialize state
#         grid = UnitGrid([64, 64])  # Consider reducing grid size for less memory
#         u = ScalarField(grid, a, label="Field $u$")
#         v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
#         state = FieldCollection([u, v])


#         process = psutil.Process()
#         memory_mb = process.memory_info().rss / 1024 / 1024
#         print(f"AA Before Memory usage: {memory_mb:.2f} MB")


#         sol = eq.solve(state, t_range=20, dt=1e-3)

  

#         process = psutil.Process()
#         memory_mb = process.memory_info().rss / 1024 / 1024
#         print(f"BB After Memory usage: {memory_mb:.2f} MB")


#         sol_tensor = torch.stack(
#             (torch.from_numpy(sol[0].data), torch.from_numpy(sol[1].data))
#         )
#         sol_tensor[~torch.isfinite(sol_tensor)] = 1e4 * torch.randn_like(
#             sol_tensor[~torch.isfinite(sol_tensor)]
#         )


#         del sol
#         gc.collect()

#         del u, v, state,eq,grid
#         gc.collect()

#         trackers.clear_cache()



#         if torch.cuda.is_available():
            
#             torch.cuda.empty_cache()
#         else:
#             print("nooo")

#         gc.collect()

#         return sol_tensor

    
#     def evaluate_true(self, X: Tensor) -> Tensor:
#         # Use torch.no_grad() to avoid unnecessary gradient tracking
#         process = psutil.Process()
#         memory_mb = process.memory_info().rss / 1024 / 1024
#         print(f"Before Memory usage: {memory_mb:.2f} MB")
    
#         with torch.no_grad():
#             # Evaluate the simulator on each of the inputs in batch
#             sims = torch.stack([self.Simulator(x) for x in X]).to(X.device)

#         # Extract the size of the grid in the simulator
#         sz = sims.shape[-1]

#         # Create a weighting array where the edges have a weight of 1/10
#         # and the middle has a weight of 1
#         weighting = (
#             torch.ones(2, sz, sz, device=sims.device, dtype=sims.dtype) / 10
#         )
#         weighting[:, [0, 1, -2, -1], :] = 1.0
#         weighting[:, :, [0, 1, -2, -1]] = 1.0

#         # Apply the weighting to the simulator outputs
#         sims = sims * weighting

#         process = psutil.Process()
#         memory_mb = process.memory_info().rss / 1024 / 1024
#         print(f"Memory usage: {memory_mb:.2f} MB")

#         print('shape: ',sims.shape)

#         # Calculate the variance of the weighted simulator outputs
#         res = 100 * sims.var(dim=(-1, -2, -3)).detach()

#         # Release memory explicitly by deleting large tensors and collecting garbage
#         del sims
#         gc.collect()

#         return res



# import psutil
# import os
# import torch
# from botorch.test_functions.base import BaseTestProblem
# from torch import Tensor
# import gc
# import tempfile
# from multiprocessing import Process
# try:
#     from pde import PDE, FieldCollection, ScalarField, UnitGrid
# except ImportError:
#     pass


# class PDEVar(BaseTestProblem):
#     def __init__(self, noise_std=None, negate=False, aggregate=False):
#         self.dim = 4
#         self._bounds = [
#             [0.1, 5.0],
#             [0.1, 5.0],
#             [0.01, 5.0],
#             [0.01, 5.0],
#         ]
#         self.num_objectives = 1
#         super().__init__(negate=negate, noise_std=noise_std)

#     @staticmethod
#     def simulator_process(tensor, temp_file):
#         # Run the simulation
#         torch.manual_seed(1234)
#         a = tensor[0].item()
#         b = tensor[1].item()
#         d0 = tensor[2].item()
#         d1 = tensor[3].item()

#         eq = PDE(
#             {
#                 "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
#                 "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
#             }
#         )

#         grid = UnitGrid([64, 64])  # Simulation grid
#         u = ScalarField(grid, a, label="Field $u$")
#         v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
#         state = FieldCollection([u, v])

#         sol = eq.solve(state, t_range=20, dt=1e-3)
#         sol_tensor = torch.stack(
#             (torch.from_numpy(sol[0].data), torch.from_numpy(sol[1].data))
#         )

#         # Replace invalid values with random noise
#         sol_tensor[~torch.isfinite(sol_tensor)] = 1e4 * torch.randn_like(
#             sol_tensor[~torch.isfinite(sol_tensor)]
#         )

#         # Save the result to a temporary file
#         torch.save(sol_tensor, temp_file)

#         # Clean up memory
#         del sol, u, v, state, eq, grid, sol_tensor
#         gc.collect()

#     def Simulator(self, tensor):
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             temp_file_name = temp_file.name

#         # Run the simulation in a separate process
#         process = Process(target=self.simulator_process, args=(tensor, temp_file_name))
#         process.start()
#         process.join()

#         # Load the result from the temporary file
#         sol_tensor = torch.load(temp_file_name)
#         os.remove(temp_file_name)  # Clean up the temporary file

#         return sol_tensor

#     def evaluate_true(self, X: Tensor) -> Tensor:
#         with torch.no_grad():
#             sims = torch.stack([self.Simulator(x) for x in X]).to(X.device)

#         sz = sims.shape[-1]
#         weighting = (
#             torch.ones(2, sz, sz, device=sims.device, dtype=sims.dtype) / 10
#         )
#         weighting[:, [0, 1, -2, -1], :] = 1.0
#         weighting[:, :, [0, 1, -2, -1]] = 1.0

#         sims = sims * weighting
#         res = 100 * sims.var(dim=(-1, -2, -3)).detach()

#         # Clean up memory
#         del sims
#         gc.collect()

#         process = psutil.Process()
#         memory_mb = process.memory_info().rss / 1024 / 1024
#         print(f"Memory usage: {memory_mb:.2f} MB")

#         return res




# import os
# import torch
# from botorch.test_functions.base import BaseTestProblem
# from torch import Tensor
# import gc
# import tempfile
# from multiprocessing import Process
# try:
#     from pde import PDE, FieldCollection, ScalarField, UnitGrid
# except ImportError:
#     pass


# class PDEVar(BaseTestProblem):
#     def __init__(self, noise_std=None, negate=False, aggregate=False):
#         self.dim = 4
#         self._bounds = [
#             [0.1, 5.0],
#             [0.1, 5.0],
#             [0.01, 5.0],
#             [0.01, 5.0],
#         ]
#         self.num_objectives = 1
#         super().__init__(negate=negate, noise_std=noise_std)

#     @staticmethod
#     def simulator_process(tensor, temp_file):
#         try:
#             # Run the simulation
#             torch.manual_seed(1234)
#             a = tensor[0].item()
#             b = tensor[1].item()
#             d0 = tensor[2].item()
#             d1 = tensor[3].item()

#             eq = PDE(
#                 {
#                     "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
#                     "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
#                 }
#             )

#             grid = UnitGrid([64, 64])  # Simulation grid
#             u = ScalarField(grid, a, label="Field $u$")
#             v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
#             state = FieldCollection([u, v])

#             sol = eq.solve(state, t_range=20, dt=1e-3)

#             sol_tensor = torch.stack(
#                 (torch.from_numpy(sol[0].data), torch.from_numpy(sol[1].data))
#             )

#             # Replace invalid values with random noise
#             sol_tensor[~torch.isfinite(sol_tensor)] = 1e4 * torch.randn_like(
#                 sol_tensor[~torch.isfinite(sol_tensor)]
#             )

#             # Save the result to a temporary file
#             torch.save({"success": True, "result": sol_tensor}, temp_file)

#         except Exception as e:
#             # Handle any errors and save the error message
#             torch.save({"success": False, "error": str(e)}, temp_file)

#         finally:
#             # Clean up memory
#             del sol, u, v, state, eq, grid
#             gc.collect()

#     def Simulator(self, tensor):
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             temp_file_name = temp_file.name

#         # Run the simulation in a separate process
#         process = Process(target=self.simulator_process, args=(tensor, temp_file_name))
#         process.start()
#         process.join()

#         # Load the result from the temporary file
#         result = torch.load(temp_file_name)
#         os.remove(temp_file_name)  # Clean up the temporary file

#         if result["success"]:
#             return result["result"]
#         else:
#             raise RuntimeError(f"Simulation failed with error: {result['error']}")

#     def evaluate_true(self, X: Tensor) -> Tensor:
#         with torch.no_grad():
#             sims = torch.stack([self.Simulator(x) for x in X]).to(X.device)

#         sz = sims.shape[-1]
#         weighting = (
#             torch.ones(2, sz, sz, device=sims.device, dtype=sims.dtype) / 10
#         )
#         weighting[:, [0, 1, -2, -1], :] = 1.0
#         weighting[:, :, [0, 1, -2, -1]] = 1.0

#         sims = sims * weighting
#         res = 100 * sims.var(dim=(-1, -2, -3)).detach()

#         # Clean up memory
#         del sims
#         gc.collect()


#         process = psutil.Process()
#         memory_mb = process.memory_info().rss / 1024 / 1024
#         print(f"Memory usage: {memory_mb:.2f} MB")


#         return res




import psutil
import os
import torch
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor
import gc
import tempfile
from multiprocessing import Process, Event
try:
    from pde import PDE, FieldCollection, ScalarField, UnitGrid
except ImportError:
    pass
import numpy as np

class PDEVar(BaseTestProblem):
    def __init__(self, noise_std=None, negate=False, aggregate=False):
        self.dim = 4
        self._bounds = [
            [0.1, 5.0],
            [0.1, 5.0],
            [0.01, 5.0],
            [0.01, 5.0],
        ]
        self.num_objectives = 1
        super().__init__(negate=negate, noise_std=noise_std)

    @staticmethod
    def simulator_process(tensor, temp_file, done_event):
        try:
            torch.manual_seed(1234)
            np.random.seed(42)

            a = tensor[0].item()
            b = tensor[1].item()
            d0 = tensor[2].item()
            d1 = tensor[3].item()

            eq = PDE(
                {
                    "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                    "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
                }
            )

            grid = UnitGrid([64, 64])  # Simulation grid
            u = ScalarField(grid, a, label="Field $u$")
            v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
            state = FieldCollection([u, v])

            # Solve the PDE
            sol = eq.solve(state, t_range=20, dt=1e-3)

            sol_tensor = torch.stack(
                (torch.from_numpy(sol[0].data), torch.from_numpy(sol[1].data))
            )

            # Replace invalid values with random noise
            sol_tensor[~torch.isfinite(sol_tensor)] = 1e4 * torch.randn_like(
                sol_tensor[~torch.isfinite(sol_tensor)]
            )

            # Save the result
            torch.save({"success": True, "result": sol_tensor}, temp_file)
        except Exception as e:
            # Handle any errors
            torch.save({"success": False, "error": str(e)}, temp_file)
        finally:
            # Signal that the process is done
            done_event.set()
            del sol, u, v, state, eq, grid
            gc.collect()


    def Simulator(self, tensor, timeout=90):

        torch.manual_seed(1234)
        np.random.seed(42)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name

        # Event to signal completion
        done_event = Event()

        # Run the simulation in a separate process
        process = Process(
            target=self.simulator_process, args=(tensor, temp_file_name, done_event)
        )
        process.start()

        # Wait for the process to complete or timeout
        process.join(timeout=timeout)

        # Check if the process is still alive (timed out)
        if process.is_alive():
            process.terminate()  # Forcefully terminate the process
            process.join()  # Ensure proper cleanup
            print(
                f"Simulation timed out after {timeout} seconds. Returning random fallback value."
            )
            # Return a random tensor as a fallback
            return torch.randn(1,2, 64, 64)

        # Ensure the process signals completion
        if not done_event.is_set():
            print(
                "Simulation process did not complete as expected. Returning random fallback value."
            )
            return torch.randn(1,2, 64, 64)

        # Load the result
        result = torch.load(temp_file_name)
        os.remove(temp_file_name)  # Clean up the temporary file

        if result["success"]:
            return result["result"]
        else:
            print(f"Simulation failed with error: {result['error']}. Returning random fallback value.")
            # Return a random tensor as a fallback
            return torch.randn(1,2, 64, 64)


    def evaluate_true(self, X: Tensor) -> Tensor:

        torch.manual_seed(1234)
        np.random.seed(42)
        
        with torch.no_grad():
            sims = torch.stack([self.Simulator(x) for x in X]).to(X.device)

        sz = sims.shape[-1]
        weighting = (
            torch.ones(2, sz, sz, device=sims.device, dtype=sims.dtype) / 10
        )
        weighting[:, [0, 1, -2, -1], :] = 1.0
        weighting[:, :, [0, 1, -2, -1]] = 1.0

        sims = sims * weighting
        res = 100 * sims.var(dim=(-1, -2, -3)).detach()

        # Clean up memory
        del sims
        gc.collect()


        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: {memory_mb:.2f} MB")


        return res


