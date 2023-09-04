import jax.numpy as jnp

def EqVar(x):
    n, p = x.shape
    done = jnp.array([], dtype=jnp.int32)
    S = jnp.cov(x.T)
    Sinv = jnp.linalg.inv(S)    
    for i in range(p):
        varmap = jnp.delete(jnp.array(range(p)), done)
        v = jnp.diag(jnp.linalg.inv(jnp.delete(jnp.delete(Sinv, done, axis=0), done, axis=1))).argmin()
        done = jnp.append(done, varmap[v])
    return done