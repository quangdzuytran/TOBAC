# Topological Ordering in Differentiable Bayesian Structure Learning with Acyclicity Assurance (TOBAC)
This is the implementation of our paper: Quang-Duy Tran, Phuoc Nguyen, Bao Duong, and Thin Nguyen. [Differentiable Bayesian Structure Learning with Acyclicity Assurance](https://arxiv.org/abs/2309.01392). In the 23rd IEEE International Conference on Data Mining (ICDM), 2023.

## Dependences 
The configuration for the conda environment is available at [```conda_linux.yml```](conda_linux.yml) (with CUDA) and [```conda_macos.yml```](conda_macos.yml).

## Running Experiments
To run the experiments, use the following files:
- Synthetic data (Figures 2, 3, & 4): [```test_joint.py```](test_joint.py),
- Flow cytometry data (Table I): [```test_joint_linear_sachs.py```](test_joint_linear_sachs.py), and
- Ablation study with different topological orderings (Figure 5): [```test_joint_ordering.py```](test_joint_ordering.py).

To see available configurations for each experiment, run the python file with ```-h``` or ```--help```.

The required configurations for all the experiments on synthetic data are 
- ```--n_vars```: the number of variables in the simulated data,
- ```--model``` (```linear``` or ```nonlinear```): the model for generating the data, and
- ```--n_particles```: the number of particles in Stein variational gradient descent.

## Citation
If you find our code helpful, please cite us as:
```
@inproceedings{tran2023topological,
  author = {Tran, Quang-Duy and Nguyen, Phuoc and Duong, Bao and Nguyen, Thin},
  booktitle = {Proceedings of the 23rd IEEE International Conference on Data Mining (ICDM)},
  title = {Differentiable {B}ayesian Structure Learning with Acyclicity Assurance},
  year = {2023}
}
```
