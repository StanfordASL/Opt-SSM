# Opt-SSM 🤖🚀
*A toolbox for learning optimal projections onto spectral submanifolds*

This codebase accompanies the paper **_Taming High-Dimensional Dynamics: Learning Optimal Projections onto Spectral Submanifolds_** ([arXiv:2504.03157](https://arxiv.org/abs/2504.03157))

<p align="center">
  <img src="./data/assets/foliation.png" width="500" alt="Optimal projection onto SSMs can improve over projecting orthogonally!">
</p>

## 🛠️ Installation

It is recommended to work in a virtual environment.

First clone the repository:
```bash
git clone git@github.com:StanfordASL/Opt-SSM.git
cd Opt-SSM
```
Install the required package dependencies and the package itself in editable mode using pip:
```bash
pip install -r requirements.txt
pip install -e .
```
Note that SSMLearnPy is currently only compatible with NumPy 1.24.3, so if you have a different version of NumPy installed, you may need to downgrade it.
Additionally, we make use of IPOPT, which has to be installed.
In case you are using conda, this can be installed using:
```bash
conda install conda-forge::ipopt
```
Finally, we recommend using Gurobi for the solving the MPC, which has to be installed separately.
However, one can also use OSQP, which can be set in the `run_mpc` function in the [mpc](./opt_ssm/mpc.py) file.

## 📘 Usage
Example usage of the OptSSM class:

```python
from utils.ssm import OptSSM

opt_ssm = OptSSM(aut_trajs_obs=aut_trajs_obs,
                 t_split=3.5,
                 SSMDim=5,
                 SSMOrder=2,
                 ROMOrder=2,
                 N_delay=3,
                 ts=ts)
```
Want to see a complete example? Check out one of the notebooks in the [examples](./examples/) directory, e.g. the [slow-fast example](./examples/slow_fast.ipynb).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

* We thank the authors of the package [SSMLearnPy](https://github.com/haller-group/SSMLearnPy) for inspiration
* Slow-fast and simplified fluid benchmarks are adapted from *Learning nonlinear projections for reduced-order modeling of dynamical systems using constrained autoencoders* ([Chaos, 2023](https://pubs.aip.org/aip/cha/article/33/11/113130/2923554))

## 📚 Citation
```
@article{buurmeijer2025optssm,
      author = {Buurmeijer, Hugo and Pabon, Luis and Alora, John Irvin and Roshan, Kaundinya and Haller, George and Pavone, Marco},
      title = {Taming High-Dimensional Dynamics: Learning Optimal Projections onto Spectral Submanifolds},
      year = {2025},
      journal = {arXiv preprint arXiv:2504.03157},
      note = {Submitted to IEEE Conference on Decision and Control (CDC), Rio de Janeiro, 2025},
      }
```
