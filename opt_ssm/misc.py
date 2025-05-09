"""
Miscellaneous utility functions.
"""

import jax
import jax.numpy as jnp
from functools import partial
from scipy import sparse
from itertools import combinations_with_replacement
import numpy as np
import osqp


class Polyhedron:
    """
    Polyhedron class.
    """
    def __init__(self, A, b, with_reproject=False):
        self.A = A
        self.b = b

        self.with_reproject = with_reproject

        if self.with_reproject:
            P = sparse.eye(self.A.shape[1]).tocsc()
            q = np.ones(self.A.shape[1])
            u = self.b
            A = sparse.csc_matrix(self.A)
            l = -np.inf * np.ones_like(self.b)
            self.osqp_prob = osqp.OSQP()
            self.osqp_prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
            self.osqp_prob.solve()

    def contains(self, x):
        """
        Returns true if x is contained in the Polyhedron.
        """
        if np.max(self.A @ x - self.b) > 0:
            return False

        else:
            return True

    @partial(jax.jit, static_argnums=(0,))
    def get_constraint_violation(self, x):
        """
        Returns distance to constraint, i.e. how large the deviation is.
        """
        return jnp.linalg.norm(jnp.maximum(self.A @ x - self.b, 0))

    def project_to_polyhedron(self, x):
        if not self.with_reproject:
            raise RuntimeError('Reproject not specified for class instance, set with_reproject=True to enable'
                               'reprojection to the Polyhedron through a QP')
        self.osqp_prob.update(q=-x)
        results = self.osqp_prob.solve()
        x_proj_alt = results.x
        return x_proj_alt

class HyperRectangle(Polyhedron):
    """
    Hyperrectangle class.
    """
    def __init__(self, ub, lb):
        n = len(ub)
        A = jnp.block(jnp.kron(jnp.eye(n), jnp.array([[1], [-1]])))
        b = jnp.hstack([jnp.array([ub[i], -lb[i]]) for i in range(n)])
        super(HyperRectangle, self).__init__(A, b)


def newton(f, x_0, tol=1e-5, max_iter=15):
    """
    Newton's method for solving f(x) = 0.
    """
    f_jac = jax.jacobian(f)

    def body_fun(val):
        x, n, error = val
        y = x - jnp.linalg.solve(f_jac(x), f(x))
        error = jnp.linalg.norm(x - y)
        return (y, n + 1, error)

    def cond_fun(val):
        _, n, error = val
        return (error > tol) & (n < max_iter)

    init_val = (x_0, 0, tol + 1)
    x_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)
    return x_final


def RK4_step(f, x, u, dt=0.01):
    """
    Perform a single step of the RK4 integration.
    """
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def fit_linear_regression(A, b, lam=0.0):
    """
    Solve normal equation for linear regression of A @ x = b.
    """
    return jnp.linalg.inv(A.T @ A + lam*jnp.eye(A.shape[-1])) @ A.T @ b


def interp_multi_output(t, ts, us):
    """
    Interpolate multiple outputs with the same time vector.
    """
    return jnp.array([jnp.interp(t, ts, u) for u in us])


def delay_embedding(input_data, up_to_delay, skips=0):
    """
    Delay embedding of input data, with latest first, followed by previous data.
    """
    undelayed_data = input_data
    buf = [undelayed_data]
    for delta in range(1, up_to_delay+1):
        skip_step = delta * (skips + 1)
        delayed_by_delta = jnp.roll(undelayed_data, -skip_step)
        delayed_by_delta = delayed_by_delta.at[:, -delta:].set(0)
        buf.append(delayed_by_delta)
    delayed_data = jnp.vstack(buf[::-1])
    return delayed_data


def trajectories_delay_embedding(input_trajs, up_to_delay, skips=0):
    """
    Delay embedding of trajectories.
    """
    N_traj, N_input_states, N_t = input_trajs.shape
    delayed_trajs = jnp.zeros((N_traj, (up_to_delay+1)*N_input_states, N_t))
    for traj in range(N_traj):
        delay_embedded_traj = delay_embedding(input_trajs[traj, :, :], up_to_delay, skips)
        delayed_trajs = delayed_trajs.at[traj, :, :].set(delay_embedded_traj)
    return delayed_trajs


def trajectories_derivatives(trajs, time):
    """
    Approximate the derivatives of trajectories with respect to time using higher-order central differences.
    """
    derivatives = jnp.zeros_like(trajs)
    dt = time[1] - time[0]

    # Fourth-order central differences for interior points
    for i in range(2, len(time) - 2):
        derivatives = derivatives.at[:, :, i].set(
            (-trajs[:, :, i + 2] + 8 * trajs[:, :, i + 1] - 8 * trajs[:, :, i - 1] + trajs[:, :, i - 2]) / (12 * dt))

    # Lower-order differences for boundaries
    derivatives = derivatives.at[:, :, 0].set((trajs[:, :, 1] - trajs[:, :, 0]) / dt)
    derivatives = derivatives.at[:, :, 1].set((-trajs[:, :, 2] + 4 * trajs[:, :, 1] - 3 * trajs[:, :, 0]) / (2 * dt))
    derivatives = derivatives.at[:, :, -2].set((3 * trajs[:, :, -1] - 4 * trajs[:, :, -2] + trajs[:, :, -3]) / (2 * dt))
    derivatives = derivatives.at[:, :, -1].set((trajs[:, :, -1] - trajs[:, :, -2]) / dt)

    return derivatives


def polynomial_features(x, degree=1, start_degree=0):
    """
    Generate polynomial features for input array x from start_degree up to a given degree,
    including interaction features.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n_samples, n_features = x.shape
    features = []

    # Add bias (ones) term if start_degree is 0
    if start_degree == 0:
        features.append(jnp.ones((n_samples, 1)))

    # Iterate over each degree from start_degree to the given degree
    for d in range(max(1, start_degree), degree + 1):
        for item in combinations_with_replacement(range(n_features), d):
            # Multiply the features corresponding to each combination
            feature = x[:, item[0]]
            for i in item[1:]:
                feature = feature * x[:, i]
            features.append(feature[:, None])  # ensure feature is a 2D array

    return jnp.concatenate(features, axis=1) if features else jnp.ones((n_samples, 1))


def compute_rmse(ground_truth, predictions, norm_axis=1, mean_axis=-1):
    """
    Compute the root mean squared error (RMSE).
    """
    return jnp.sqrt(jnp.mean(jnp.linalg.norm(ground_truth - predictions, axis=norm_axis) ** 2, axis=mean_axis))


def get_min_max(values, margin):
    """
    Get the min and max values from a list of lists of values.
    """
    flat_values = [item for sublist in values for item in sublist]
    return min(flat_values) - margin, max(flat_values) + margin
