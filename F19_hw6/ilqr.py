"""LQR, iLQR and MPC."""

from controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg
import pdb

def simulate_dynamics_next(env, x, u,dt=1e-5):
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
    x_new = env.step(u,dt)[0]
    
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
    x_goal = np.concatenate((env.goal_q,env.goal_dq),0)
    
    # l = np.matmul(np.matmul((x - x_goal).T,env.Q),(x-x_goal))
    # l += np.matmul(np.matmul(u.T,env.R),u)
    # l_x = 2*np.matmul(env.Q,(x-x_goal))
    # l_xx = 2*env.Q*1.0
    # l_u = 2*np.matmul(env.R,u)
    # l_uu = 2*env.R*1.0
    
    l = np.sum(u**2)
    l_x = 0*2*np.matmul(env.Q,(x-x_goal))
    l_xx = 0*2*env.Q*1.0
    l_u = 2*u
    l_uu = 2*np.eye(u.shape[0])
    
    l_ux = np.zeros((u.shape[0],x.shape[0]))
    return l,l_x,l_xx, l_u, l_uu, l_ux


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

    x_goal = np.concatenate((env.goal_q,env.goal_dq),0)
    # l = 10*np.matmul(np.matmul((x - x_goal).T,env.Q),(x-x_goal))

    # l_x = 10*2*np.matmul(env.Q,(x-x_goal))
    # l_xx = 2*env.Q*10

    l_fac = 1e4
    l = l_fac*np.sum((x-x_goal)**2)
    l_x = l_fac*2*(x-x_goal)
    l_xx = l_fac*2*np.eye(x.shape[0])
    
    return l,l_x,l_xx


def simulate(env, x0, U):
    env.state = x0*1.0
    x_seq = []
    done = False
    max_steps = U.shape[0]
    num_steps = 0
    cost = 0
    x = x0*1.0
    while (not done) and (num_steps<max_steps):
        x_seq.append(x*1.0)
        x_next,reward,done,info = env.step(U[num_steps])
        x = x_next*1.0 
        num_steps += 1
        cost -= reward
    x_seq.append(x*1.0)
    return np.array(x_seq), cost


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
    
    x_seq, cost = simulate(sim_env,x0,u_seq)
    num_iter = 0
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    lamb = 1.0
    lamb_fac = 2.0
    lmab_max = 2000
    alpha = 1.0
    
    lamb_min = 1e-6
    del_lamb0 = 2
    lamb = lamb_min*1.0
    del_lamb = 0
    while num_iter < max_iter:
        num_iter += 1
        print("iter: {}".format(num_iter))
        env.state = x0*1.0
        l = np.zeros((tN+1,1))
        l_x = np.zeros((tN+1,state_dim))
        l_xx = np.zeros((tN+1,state_dim,state_dim))
        l_u = np.zeros((tN,action_dim))
        l_uu = np.zeros((tN,action_dim,action_dim))
        l_ux = np.zeros((tN,action_dim,state_dim))
        f_x = np.zeros((tN,state_dim,state_dim))
        f_u = np.zeros((tN,state_dim,action_dim))


        l[-1],l_x[-1],l_xx[-1] = cost_final(sim_env,x_seq[-1])
        for i in range(tN-1,-1,-1):
            x = x_seq[i] * 1.0
            u = u_seq[i] * 1.0
            
            l[i],l_x[i],l_xx[i],l_u[i],l_uu[i],l_ux[i] = cost_inter(sim_env,x,u)
         
            A = approximate_A(sim_env,simulate_dynamics_next,x,u)
            B = approximate_B(sim_env,simulate_dynamics_next,x,u)
            
            f_x[i] = A*1.0
            f_u[i] = B*1.0
        k = np.zeros((tN,action_dim))
        K = np.zeros((tN,action_dim,state_dim))
        
        # k: open loop term; K: feedback gain term
        pd_flag = 0
        lamb /= lamb_fac
        while pd_flag==0:
            print(pd_flag)
            lamb *= lamb_fac
            k,K,pd_flag = backward_recursion(k,K,l,l_x,l_xx,l_u,l_uu,l_ux,f_x,f_u,lamb)
            print(pd_flag)
        
        lamb /= lamb_fac
        u_seq_new = np.zeros((tN,action_dim))

        x_new = x_seq[0] * 1.0
        sim_env.state = x_new*1.0
        

        for i in range(tN-1):
            u_seq_new[i] = u_seq[i] + alpha*k[i] + np.matmul(K[i],x_new - x_seq[i])
            x_new,_,_,_ = sim_env.step(u_seq_new[i])
            # x_new = simulate_dynamics_next(sim_env,x_new,u_seq_new[i])

        x_seq_new, new_cost = simulate(env,x0,u_seq_new)

        u_seq = u_seq_new*1.0
        # x_seq = x_seq_new*1.0
        print("new_cost: {}".format(new_cost))
        # if new_cost < cost:
            # del_lamb = min(1/del_lamb0,del_lamb/del_lamb0)
            # lamb = lamb*del_lamb if lamb*del_lamb>lamb_min else 0
            # # lamb /= lamb_fac
            # x_seq = x_seq_new * 1.0
            # u_seq = u_seq_new * 1.0
            # alpha = 1.0 
            # if(abs(cost-new_cost) < 0.001*cost):
                # print("new_cost: {}".format(new_cost))
                # cost = new_cost * 1.0
                # break
            # cost = new_cost*1.0
        
        # else:
            # del_lamb = max(del_lamb0,del_lamb*del_lamb0)
            # lamb = max(lamb_min,lamb*del_lamb)
            # # lamb *= lamb_fac
            # # alpha /= 10
            # print(lamb)
            # if lamb > lmab_max:
                # print("cost: {}".format(cost))     
                # break
    return u_seq



def inv_stable(M,lamb=1):
    M_evals, M_evecs = np.linalg.eig(M)
    M_evals[M_evals<0] = 0.0
    M_evals += lamb
    M_inv = np.dot(M_evecs, np.dot(np.diag(1.0/M_evals),M_evecs.T))
    
    return M_inv


def backward_recursion(k,K,l,l_x,l_xx,l_u,l_uu,l_ux,f_x,f_u,lamb):
    V = l[-1]*1.0
    V_x = l_x[-1]*1.0
    V_xx = l_xx[-1]*1.0
    tN = k.shape[0]
    pd_flag = 0
    for i in range(tN-1,-1,-1):
        Q_x = l_x[i] + np.matmul(f_x[i].T,  V_x)
        Q_u = l_u[i] + np.matmul(f_u[i].T, V_x)
        
        Q_xx = l_xx[i] + np.matmul(f_x[i].T, np.matmul(V_xx,f_x[i]))
        Q_uu = l_uu[i] + np.matmul(f_u[i].T, np.matmul(V_xx,f_u[i]))
        Q_ux = l_ux[i] + np.matmul(f_u[i].T, np.matmul(V_xx,f_x[i]))
        
        try:
            _ = np.linalg.cholesky(Q_uu)
            pd_flag = 1
        except:
            pd_flag = 0
            break

        Q_uu_inv = inv_stable(Q_uu,lamb=lamb)
        # Q_uu_inv = np.linalg.pinv(Q_uu) 
        k[i] = -np.matmul(Q_uu_inv,Q_u)
        K[i] = -np.matmul(Q_uu_inv,Q_ux)

        # V_x = Q_x - Q_u @ K[i]
        # V_xx = Q_xx - Q_ux.T @ K[i]
        

        V_x = Q_x + Q_ux.T @ k[i]
        V_xx = Q_xx + Q_ux.T @ K[i]

        # V_x = Q_x - np.matmul(K[i].T, np.matmul(Q_uu,k[i]))
        # V_xx = Q_xx - np.matmul(K[i].T,np.matmul(Q_uu, K[i]))

    return k,K,pd_flag







