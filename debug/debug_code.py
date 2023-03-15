import numpy as np
from receding_horizon_control import receding_horizon_plan
import pickle

if __name__ == '__main__':
    start_loc = np.array([0, 0])
    goal_loc = np.array([8, 9])
    grid_size = (10, 15)
    n_obst = 30
    edge_size = 1
    gamma = 0.1
    
    env = pickle.load(open('debug/env_rev', 'rb'))
    W = pickle.load(open('debug/W_rev', 'rb'))
    
    num_disc = 50
    num_gamma = 10
    num_draws = 10
    gamma_vals = np.linspace(0.01, 0.5, num_disc, endpoint=True)
    # Uniform initialization of gamma distribution i.e. uniform prior on gamma
    # Use log pdf
    prior_dist = np.log(1/num_disc * np.ones(num_disc))
            
    traj, W_list, gamma_beliefs = receding_horizon_plan(start_loc, goal_loc, W, 
                                gamma_vals, prior_dist, env, num_gamma, num_draws, True)