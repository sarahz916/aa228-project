import numpy as np    
import pickle
import matplotlib.pyplot as plt
from receding_horizon_control import receding_horizon_plan
from env_gen import create_feasible_occ
from env import Environment
from create_wind_vector import create_wind_vector
from find_one_path import find_best_path, path_to_moves

def suboptimality_study(n_samples, gamma_range, task_creator, planner):
    """Characterize suboptimality as a function of gamma.""" 
    
    feas_costs = np.zeros((len(gamma_range), n_samples))
    actual_costs =  np.zeros((len(gamma_range), n_samples))
    best_costs = np.zeros((len(gamma_range), n_samples))
    
    for i, g in enumerate(gamma_range):
        print('On gamma = ' + str(g))
        for j in range(n_samples):
            print('On sample ' + str(j))
            
            start_loc, goal_loc, env, feas_path = task_creator()
            
            W = create_wind_vector(env.occ.shape, g)[0]
            
            pickle.dump(env, open('debug/env', 'wb'))
            pickle.dump(W, open('debug/W', 'wb'))
            
            traj = planner(start_loc, goal_loc, env, W)
            
            _, best_path, best_cost, _ = find_best_path(start_loc, goal_loc, env, [W])
            feas_cost = env.path_cost(start_loc, path_to_moves(feas_path), W)[0]
            traj_cost = env.path_cost(start_loc, path_to_moves(traj), W)[0]
            
            feas_costs[i,j] = feas_cost
            actual_costs[i,j] = traj_cost
            best_costs[i,j] = best_cost
    
    return feas_costs, actual_costs, best_costs

if __name__ == '__main__':
    
    # Initialize the task creator
    start_loc = np.array([0, 0])
    goal_loc = np.array([8, 9])
    grid_size = (10, 15)
    n_obst = 30
    def task_creator():
        feas_path, occ = create_feasible_occ(n_obst, grid_size, start_loc, goal_loc)
        env = Environment(occ, 1, 0.1, buildings_block=True)  
        return start_loc, goal_loc, env, feas_path
    
    # Initialize the planner
    num_disc = 50
    gamma_vals = np.linspace(0.01, 0.5, num_disc, endpoint=True)
    prior_dist = np.log(1/num_disc * np.ones(num_disc))
    num_gamma = 10
    num_draws = 10 
    def planner(start_loc, goal_loc, env, W):
        return receding_horizon_plan(start_loc, goal_loc, W, gamma_vals, 
            prior_dist, env, num_gamma, num_draws)[0]
    
    n_samples = 50
    gamma_range = np.linspace(0.01, 0.5, 10)
    feas_costs, actual_costs, best_costs = suboptimality_study(n_samples, 
                                            gamma_range, task_creator, planner)
        
    # Compare the method against oracle and baseline
    fig = plt.figure()
    names = ['Feasible', 'Actual', 'Best']
    datasets = [feas_costs, actual_costs, best_costs]
    for i, name in enumerate(names):
        data = datasets[i] # should have shape (len(gamma_range), n_samples)
        medians = np.median(data, axis=1) # should have shape len(gamma_range)
        # quartiles = np.quantile(data, [0.25, 0.75], axis=1) # should have shape 2xN lower, upper
        # Make them offsets
        # offsets = np.copy(quartiles)
        # offsets[0,:] -= medians 
        # offsets[0,:] *= -1
        # offsets[1,:] -= medians
        offsets = np.std(data, axis=1) / np.sqrt(n_samples) 
        plt.errorbar(gamma_range, medians, yerr=offsets, label=name, capsize=10)
        
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Trajectory Cost')
    plt.title(r'Comparing Trajectory Costs Across $\gamma$')
    plt.legend()


