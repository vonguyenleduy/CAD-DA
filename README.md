# CAD-DA: Controllable Anomaly Detection after Domain Adaptation by Statistical Inference (AISTATS 2024)

This package implements a novel statistical method for testing the results of anomaly detection (AD) under domain adaptation (DA), which we call CAD-DA—controllable AD under DA.

See the paper <https://arxiv.org/abs/2310.14608> for more details.

## Installation & Requirements

This package has the following requirements:

- [numpy](http://numpy.org)
- [mpmath](http://mpmath.org/)
- [matplotlib](https://matplotlib.org/)
- [scipy](https://www.scipy.org)
- [statsmodels](https://www.statsmodels.org/stable/index.html)

We recommend to install or update anaconda to the latest version and use Python 3
(We used Python 3.8.3).

**NOTE: We use scipy package to solve the linear program (simplex method). However, the default package does not return the set of basic variables. Therefore, we slightly modified the package so that it can return the set of basic variables by replacing the two files '_linprog.py' and '_linprog_simplex.py' in scipy.optimize module with our modified files in the folder 'file_to_replace'.**

**NOTE: Please follow the above replacing step. Otherwise, you can not run the code.**

### Example of computing the p-value
```
>> python example_computing_p_value.py
```

