import numpy as onp
import jax
import jax.numpy as np
from jax import grad, jit, vmap


# TODO...

@jax.jit
def adjoint_rhs_func(params, state, adjoint):
    jac_f_u = jac_rhs_state(params, state)
    jac_f_u = jac_f_u.reshape(state.size, state.size)
    adjoint = adjoint.reshape(state.size)
    result = -adjoint @ jac_f_u
    return result.reshape(state.shape)


def solve_adjoints(params, states, num_steps, dt):
    adjoint = grad_objective_state(params, states[-1])
    adjoints = [adjoint]
    for i in range(num_steps):
        state = states[num_steps - i]
        rhs_func = lambda variable: adjoint_rhs_func(params, state, variable)
        adjoint = explicit_euler(adjoint, rhs_func, -dt)         
        adjoints.append(adjoint)
        # if i % 20 == 0:
        #     print(f"Adjoint ODE \nstep {i}")
    return np.flip(np.array(adjoints), axis=0)