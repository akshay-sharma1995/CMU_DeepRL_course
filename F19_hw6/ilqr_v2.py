"""LQR, iLQR and MPC."""

from controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg
import pdb

def simulate_dynamics_next(env, x, u, dt=1e-5):
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

    Returns
    -------
    next_x: np.array
    """
    env.state = x*1.0
    x_new,_,_,_ = env.step(u,dt)
    
    return x_new


def cost_inter(env, x, u):
    """intermediate cost function

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

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    x_goal = env.goal*1.0

    l = np.sum(u**2) ## cost
    l_x = np.zeros_like(x) ## d(l)/dx
    l_xx = np.zeros((x.shape[0],x.shape[0])) ## d2(l)/dx2

    l_u = 2*u
    l_uu = 2*np.eye(u.shape[0])
    l_ux = np.zeros((u.shape[0],x.shape[0]))

    return l, l_x, l_xx, l_u, l_uu, l_ux


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    x_goal = env.goal*1.0

    l_factor = 1e4

    l = l_factor * np.sum((x - x_goal)**2)
    l_x = l_factor * 2 * (x - x_goal)
    l_xx = l_factor * 2 * np.eye(x.shape[0])

    return l, l_x, l_xx

def simulate(env, x0, U, render=False):

    env.state = x0 * 1.0
    traj_horizon = U.shape[0]
    x_goal = env.goal*1.0
    
    cost = 0
    x_seq = np.zeros((traj_horizon+1 , *x0.shape))
    x_seq[0] = x0*1.0

    t = 0
    done = False
    
    total_reward = 0

    while (not done) and (t < traj_horizon):
        if render:
            env.render()
        x_next, reward, done , info = env.step(U[t])
        x_seq[t+1] = x_next * 1.0
        cost += np.sum(U[t]**2)
        t += 1
        total_reward += reward*1.0
    cost += 1e4 * np.sum((x_next - x_goal)**2)
    
    return x_seq, cost, total_reward


# def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6):
def calc_ilqr_input(env, sim_env,x0, u_seq, tN=50, max_iter=1e6):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    
    state_dim = x0.shape[0]
    action_dim = u_seq.shape[1]
    
    lamb = 1.0
    lamb_fac = 2.0
    lamb_max = 1000

    iter_num = 0
    gen_new_traj = True 


    while(iter_num < max_iter):
    
        if gen_new_traj:
            pdb.set_trace()
            x_seq, cost, reward = simulate(sim_env, x0, u_seq)
            old_cost = cost * 1.0
            env.state = x0 * 1.0
            
            l, l_x, l_xx, l_u, l_uu, l_ux, f_x, f_u = reset_cost_dynamics(tN,state_dim, action_dim)

            l[-1], l_x[-1], l_xx[-1] = cost_final(sim_env, x_seq[-1])
            
            for i in range(tN-1, -1, -1):
                x = x_seq[i] * 1.0
                u = u_seq[i] * 1.0

                l[i], l_x[i], l_xx[i], l_u[i], l_uu[i], l_ux[i] = cost_inter(sim_env, x, u)
                 
                A = approximate_A(sim_env, simulate_dynamics_next, x, u)
                B = approximate_B(sim_env, simulate_dynamics_next, x, u)

                f_x[i] = A * 1.0
                f_u[i] = B * 1.0

                gen_new_traj = False
            
            pdb.set_trace()
            k = np.zeros((tN, action_dim))
            K = np.zeros((tN, action_dim, state_dim))

            k,K = backward_recursion(k, K, l, l_x, l_xx, l_u. l_uu, l_ux, f_x, f_u , lamb)

    return np.zeros((50, 2))


def reset_cost_dynamics(tN,state_dim,action_dim):
    l = np.zeros((tN+1,))
    l_x = np.zeros((tN+1,state_dim))
    l_xx = np.zeros((tN+1,state_dim,state_dim))
    l_u = np.zeros((tN,action_dim))
    l_uu = np.zeros((tN,action_dim,action_dim))
    l_ux = np.zeros((tN,action_dim,state_dim))
    f_x = np.zeros((tN,state_dim,state_dim))
    f_u = np.zeros((tN,state_dim,action_dim))
    
    return l,l_x,l_xx,l_u,l_uu,l_ux,f_x,f_u




