"""LQR, iLQR and MPC."""

import numpy as np
import pdb
import scipy
import scipy.linalg
import pdb

def simulate_dynamics(env, x, u, dt=1e-5):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    env.state = x*1.0
    x_new = env.step(u,dt)[0]
    xdot = (x_new - x) / dt
    return xdot

def approximate_A(env,simulate_dynamics, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """

    # linearization point
    x0 = x*1.0
    u0 = u*1.0
    env.state = x0*1.0
    
    f0 = simulate_dynamics(env,x0,u0)

    A = np.zeros((x.shape[0],x.shape[0]))
    for i in range(x0.shape[0]):
        x_next = x0*1.0
        x_prev = x0*1.0
        x_next[i] += delta
        x_prev[i] -= delta
        f_next = simulate_dynamics(env,x_next,u0)
        f_prev = simulate_dynamics(env,x_prev,u0)
        A[:,i] = (f_next - f_prev) / (2*delta)

    return A

def approximate_B(env,simulate_dynamics, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    # linearization point
    x0 = x*1.0
    u0 = u*1.0
    env.state = x0*1.0
    f0 = simulate_dynamics(env,x0,u0,dt)
    
    B = np.zeros((x.shape[0],u.shape[0]))

    for i in range(u.shape[0]):
        u_next = u0*1.0
        u_prev = u0*1.0
        u_next[i] += delta
        u_prev[i] -= delta

        env.state = x0*1.0
        f_next = simulate_dynamics(env,x0,u_next)
        f_prev = simulate_dynamics(env,x0,u_prev)

        B[:,i] = (f_next - f_prev) / delta
    
    return B


# def calc_lqr_input(env, sim_env):
def calc_lqr_input(env,x,u):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """ 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    x_goal = np.concatenate((env.goal_q,env.goal_dq),0)
    u_goal = np.zeros((2,))

    A = approximate_A(env,simulate_dynamics,x,u,delta=1e-5,dt=1e-5)
    B = approximate_B(env,simulate_dynamics,x,u,delta=1e-5,dt=1e-5)
    
    Q = env.Q
    R = env.R
    
    P = scipy.linalg.solve_continuous_are(A,B,Q,R) 
    R_inv = scipy.linalg.inv(R)

    K = np.matmul(np.matmul(R_inv,B.T) , P)
    u = -np.matmul(K,(x-x_goal))
    
    return u





