# Respecting the limit: Bayesian optimization with a bound on the optimal value
In many real-world optimization problems, we have prior information about what objective
function values are achievable. In this paper, we study the scenario that we have either
exact knowledge of the minimum value or a, possibly inexact, lower bound on its value. We
propose bound-aware Bayesian optimization (BABO), a Bayesian optimization method that
uses a new surrogate model and acquisition function to utilize such prior information. We
present SlogGP, a new surrogate model that incorporates bound information and adapts the
Expected Improvement (EI) acquisition function accordingly. Empirical results on a variety
of benchmarks demonstrate the benefit of taking prior information about the optimal value
into account, and that the proposed approach significantly outperforms existing techniques.
Furthermore, we notice that even in the absence of prior information on the bound, the
proposed SlogGP surrogate model still performs better than the standard GP model in
most cases, which we explain by its larger expressiveness.

# How to run
Here is how to use this package:
```bash
git clone https://github.com/HanyangHenry-Wang/BABO.git && cd BABO
conda create --name babo-env python=3.9.16 -y
conda activate babo-env
pip install -r requirements.txt 
```
Then we can run the experiment:
```bash
python run_experiment.py
```
Details can be adjusted in run_experiment.py.

# OBCGP
We have also re-implemented the OBCGP algorithm, which handles Bayesian Optimization with a known bound. Here is the link: https://github.com/HanyangHenry-Wang/OBCGP_rewrite.git
