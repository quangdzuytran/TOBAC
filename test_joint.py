from tobac.inference_dibs import JointDiBS
from tobac.inference_ours import JointTOBAC
from tobac.target import make_linear_gaussian_model, make_nonlinear_gaussian_model
from tobac.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from tobac.graph_utils import elwise_acyclic_constr_nograd
import jax
import jax.random as random
from itertools import product
import pandas as pd
from argparse import ArgumentParser

models = {
    "linear": make_linear_gaussian_model,
    "nonlinear": make_nonlinear_gaussian_model,
}

methods = {
   "DiBS": JointDiBS,
   "OURS": JointTOBAC
}

def process(task):
    taskid, (subk, data_config, method, n_particles, steps) = task
    graph_prior_str = data_config["graph_prior_str"]
    n_vars = data_config["n_vars"]
    edges_per_node = data_config["edges_per_node"]
    n_obs = data_config["n_obs"]

    data, model = models[data_config["model"]](
        key=subk, 
        n_vars=n_vars, 
        graph_prior_str=graph_prior_str,
        edges_per_node=edges_per_node,
        n_observations=n_obs)
    
    obj = methods[method["name"][:4]](
        x=data.x, 
        interv_mask=None, 
        inference_model=model)
    if method["gt_perm"]:
        obj.perm = data.perm
    
    gs, thetas = obj.sample(key=subk,
                            n_particles=n_particles, 
                            steps=steps,
                            )
    
    emprical = obj.get_empirical(g=gs, theta=thetas)
    mixture = obj.get_mixture(g=gs, theta=thetas)
    n_cyclic = (elwise_acyclic_constr_nograd(gs, n_vars) > 0).sum().item()
    results = []
    for dist_name, dist in [("Empirical", emprical), ("Mixture", mixture)]:
        eshd = expected_shd(dist=dist, g=data.g)
        auroc = threshold_metrics(dist=dist, g=data.g)["roc_auc"]
        negll = neg_ave_log_likelihood(dist=dist, x=data.x_ho, eltwise_log_likelihood=obj.eltwise_log_likelihood_observ)
        result_dict = {
            "Key": subk,
            "Prior": graph_prior_str,
            "Degree": edges_per_node//2,
            "N_Obs": n_obs,
            "Vars": data.n_vars,
            "Particles": n_particles,
            "Method": method["name"] + ("+" if dist_name == "Mixture" else ""),
            "Steps": steps,
            "Cyclic": n_cyclic,
            "E-SHD": float(eshd),
            "AUROC": float(auroc),
            "Neg.LL": float(negll)
        }
        results.append(result_dict)
    print("Done", taskid)
    print(results[0])
    print(results[1])
    return results

if __name__ == "__main__":
    print(f"JAX backend: {jax.default_backend()}")
    parser = ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--graph_prior', type=str, choices=['er', 'sf'], default='er')
    parser.add_argument('--edges_per_node', type=int, default=2)
    parser.add_argument('--model', type=str, required=True, choices=['linear', 'nonlinear'])
    parser.add_argument('--n_vars', type=int, required=True)
    parser.add_argument('--n_particles', type=int, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--n_obs', type=int, default=100)
    args = parser.parse_args()

    key = random.PRNGKey(args.seed)
    subkeys = list(random.split(key=key, num=args.n_runs))

    data_configs = [{"graph_prior_str": args.graph_prior, 
                        "edges_per_node": args.edges_per_node,
                        "n_vars": args.n_vars,
                        "model": args.model,
                        "n_obs": args.n_obs}]

    method_list = [{"name": "DiBS", "gt_perm": False}, 
                    {"name": "OURS_GT", "gt_perm": True},
                    {"name": "OURS_EqVar", "gt_perm": False}
                ]

    tasks = list(enumerate(product(subkeys, data_configs, method_list, [args.n_particles], [args.steps])))
    print("Number of tasks:", len(tasks))

    res = list(map(process, tasks))
    
    res = [res_dict for results in res for res_dict in results ]

    res_df = pd.DataFrame(res)
    res_df.to_csv(f"results/ours_j_{args.model[:3]}_{args.graph_prior}-{args.edges_per_node//2}_d{args.n_vars}-m{args.n_particles}-T{args.steps}_n{args.n_obs}_r{args.n_runs}_s{args.seed}_{args.exp_name}.csv")
    print(res_df)