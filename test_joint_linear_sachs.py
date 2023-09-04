import jax.numpy as jnp
import networkx as nx
import pandas as pd 
from jax import default_backend
from jax import random
from tobac.inference_dibs import JointDiBS
from tobac.inference_ours import JointTOBAC
from tobac.target import Data, make_graph_model
from tobac.models import LinearGaussian
from tobac.graph_utils import elwise_acyclic_constr_nograd
from tobac.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def nx_adjacency(g):
    return jnp.array(nx.adj_matrix(g).toarray())

def load_sachs(subk, normalize=True, held_out=True):
    passed_key = subk
    sachs_data = pd.read_csv('data/sachs_observational.csv')
    sachs_gt_graph = nx.from_numpy_matrix(jnp.load('data/sachs_ground_truth.npy'), create_using=nx.DiGraph)

    g_gt_mat = jnp.array(nx_adjacency(sachs_gt_graph))
    # g_gt_order = jnp.array(list(nx.topological_sort(sachs_gt_graph)))
    g_gt_order = None
    # g_gt_perm = jnp.vstack([jnp.zeros(n_vars).at[i].set(1) for i in g_gt_order], dtype=jnp.float32).T
    g_gt_perm = None
    theta = None

    sachs_x_full = jnp.array(pd.DataFrame(data=sachs_data.to_numpy()).values)

    if normalize:
        sachs_x_full = (sachs_x_full - sachs_x_full.mean(0, keepdims=True)) / sachs_x_full.std(0, keepdims=True)
    
    key, subk = random.split(subk)
    if held_out:
        x_full = random.permutation(subk, sachs_x_full)

        cut = int(0.9 * x_full.shape[0])
        x = x_full[:cut]
        x_ho = x_full[cut:]
        x_interv = None

        n_observations, n_vars = x.shape
        n_ho_observations, _ = x_ho.shape
    else:
        x = random.permutation(subk, sachs_x_full)

        x_ho = None
        x_interv = None

        n_observations, n_vars = x.shape
        n_ho_observations = 0
    
    data = Data(
        passed_key=passed_key,
        n_vars=n_vars,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations,
        g=g_gt_mat,
        ordering=g_gt_order,
        perm=g_gt_perm,
        theta=theta,
        x=x,
        x_ho=x_ho,
        x_interv=x_interv
    )

    graph_dist = make_graph_model(n_vars=n_vars, graph_prior_str='er')
    inference_model = LinearGaussian(
        graph_dist=graph_dist, 
        obs_noise=0.1,
        mean_edge=0.0, 
        sig_edge=1.0,
        min_edge=0.5
    )

    return data, inference_model

methods = {
   "DiBS": JointDiBS,
   "OURS": JointTOBAC
}

def process(subk, method, n_particles, steps):
    print(method["name"], n_particles, steps)
    data, model = load_sachs(subk=subk)

    n_vars = data.n_vars
    graph_prior_str = 'er'

    obj = methods[method["name"][:4]](
        x=data.x, 
        interv_mask=None, 
        inference_model=model)
    if method["gt_perm"]:
        obj.perm = data.perm
    
    gs, thetas = obj.sample(key=subk,
                            n_particles=n_particles, 
                            steps=steps,
                            # callback_every=50,
                            # callback=obj.visualize_callback(ipython=False)
                            )
    
    emprical = obj.get_empirical(g=gs, theta=thetas)
    mixture = obj.get_mixture(g=gs, theta=thetas)
    n_cyclic = (elwise_acyclic_constr_nograd(gs, n_vars) > 0).sum().item()
    results = []
    for dist_name, dist in [("Empirical", emprical), ("Mixture", mixture)]:
        eshd = expected_shd(dist=dist, g=data.g)
        auroc = threshold_metrics(dist=dist, g=data.g)["roc_auc"]
        if data.x_ho is not None:
            negll = neg_ave_log_likelihood(dist=dist, x=data.x_ho, eltwise_log_likelihood=obj.eltwise_log_likelihood_observ)
        else:
            negll = float("nan")
        result_dict = {
            "Key": subk,
            "Prior": graph_prior_str,
            "Vars": data.n_vars,
            "Particles": n_particles,
            "Method": method["name"] + ("+" if dist_name == "Mixture" else ""),
            "Steps": steps,
            "Cyclic": n_cyclic,
            "E-SHD": float(eshd),
            "AUROC": float(auroc),
            "Neg.LL": float(negll)
        }
        print("E-SHD:", eshd)
        print("AUROC:", auroc)
        print("Neg.LL", negll)
        results.append(result_dict)
    return results

if __name__ == "__main__":
    print(f"JAX backend: {default_backend()}")
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default="")
    args = parser.parse_args()
    key = random.PRNGKey(0)
    n_particles_list = [30]
    steps_list = [1000]
    n_exps = 10

    method_list = [
                    {"name": "DiBS", "gt_perm": False},
                    {"name": "OURS_EqVar", "gt_perm": False},
                    # {"name": "OURS_GT", "gt_perm": True}
                    ]
    res = []
    subkeys = list(random.split(key=key, num=n_exps))
    for subkey in subkeys:
        for n_particles in n_particles_list:
            for steps in steps_list:
                for method in method_list:
                    res += process(subk=subkey,
                            method=method,
                            n_particles=n_particles,
                            steps=steps)
    
    res_df = pd.DataFrame(res)
    res_df.to_csv(f"results/ours_lin_sachs_s{args.seed}_{args.exp_name}.csv")
    print(res_df)

        