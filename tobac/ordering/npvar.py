import jax.numpy as jnp
# pip install pygam
from pygam import GAM
from pygam.terms import s, TermList

def NPVar(X, verbose=False, method='mgcv', **kwargs):
    n, p = X.shape
    node_index = jnp.array(list(range(p)))

    condvars = jnp.var(X, axis=0)
    source = jnp.argmin(condvars)
    ancestors = jnp.array([source])
    if verbose:
        # print(dict(zip(node_index, condvars)))
        print(ancestors)
    
    while len(ancestors) < p - 1:
        descendants = jnp.delete(node_index, ancestors)
        condvars = est_condvars(X, descendants, ancestors, verbose, method)
        min_index = jnp.argmin(condvars)
        source_next = descendants[min_index]
        ancestors = jnp.append(ancestors, source_next)
        if verbose:
            # print(dict(zip(descendants, condvars)))
            print(ancestors)

    descendants = jnp.delete(node_index, ancestors)
    if len(ancestors) < p:
        ancestors = jnp.append(ancestors, descendants)
    if verbose:
        print(ancestors)
    return ancestors

def est_condvars(X, descendants, ancestors, verbose, method):
    assert method in ['np', 'mgcv']
    condvars = jnp.array([float('nan')] * len(descendants))
    for i in range(len(descendants)):
        current_node = descendants[i]
        if verbose:
            print("Checking", "X" + str(current_node), " ~ ", " + ".join(["X" + str(a) for a in ancestors]))
        if method == 'np':
            pass
        elif method == 'mgcv':
            mgcv_formula = [s(a) for a in ancestors]
            b1 = GAM(terms=TermList(*mgcv_formula)).fit(X=X, y=X[:, current_node])
            fit_gam = b1.predict(X=X)
            condvar_gam = jnp.var(X[:, current_node]) - jnp.var(fit_gam)
            condvars.at[i].set(condvar_gam)
    return condvars