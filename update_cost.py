import numpy as np
import matplotlib.pyplot as plt
from bayesian_updating import update_gamma_dist, sample_posterior_wind_vector
from env import Environment, get_colors, set_colorbar
from env_gen import vis_path, vis_map, create_feasible_occ
from create_wind_vector import create_wind_vector, plot_wind_vector
from find_one_path import find_best_path, path_to_moves, vis_path

### Plot Solver trajectory
def plot_solver():
    start_loc = np.array([0, 0])
    grid_size = (10, 15)
    goal_loc = np.array([9, 14])
    n_obst = 5
    edge_size = 1
    gamma = 0.1
    
    feas_path, occ = create_feasible_occ(n_obst, grid_size, start_loc, goal_loc)
    env = Environment(occ, 1, 0.1, buildings_block=True)  
    W, dists, locs = create_wind_vector(grid_size, gamma) 
    num_disc = 50
    gamma_vals = np.linspace(0.01, 1, num_disc, endpoint=True)
    # Uniform initialization of gamma distribution i.e. uniform prior on gamma
    # Use log pdf
    prior_dist = np.log(1/num_disc * np.ones(num_disc))
    
    num_gamma = 10
    num_draws = 10
    
    old_locs = []
    old_wind_meas = []
    sign_locs = []
    wind_signs = []
    # take measurement 
    curr_loc = start_loc
    traj = [start_loc]
    while (curr_loc != goal_loc).any():
        new_wind_meas_x, sign_x = env.perceived_wind_x(W, curr_loc)
        new_wind_meas_y, sign_y = env.perceived_wind_y(W, curr_loc)
        new_wind_meas = np.array([new_wind_meas_x, new_wind_meas_y])
        
        new_dist_rec = update_gamma_dist(gamma_vals, prior_dist, 
                                         np.expand_dims(new_wind_meas, axis=0), [curr_loc]) # TODO: add old meas
        
        if new_wind_meas_x != None or new_wind_meas_y != None: # want to add some information
            old_wind_meas.append(new_wind_meas)
            old_locs.append(curr_loc)
        if sign_x != 0 or sign_y != 0 and (new_wind_meas_x == None or new_wind_meas_y == None):
            sign_locs.append(curr_loc)
            wind_signs.append(np.array([sign_x, sign_y]))
        
        gamma_draws, W_draws = sample_posterior_wind_vector(grid_size, gamma_vals, new_dist_rec, 
                                         num_gamma, num_draws, np.array(old_wind_meas), 
                                         np.array(old_locs), np.array(wind_signs), np.array(sign_locs))
        
        
        W_list = W_draws.reshape((-1, grid_size[0], grid_size[1], 2))
        feasible, path, avg_cost, path_costs = find_best_path(curr_loc, goal_loc, env, W_list)
        moves = path_to_moves(path)
        next_loc = curr_loc + moves[0]
        traj.append(next_loc)
        curr_loc = next_loc
        
        
        
    traj = np.array(traj)
    
    fig, ax = plt.subplots()
    vis_map(occ, ax)
    plt.grid(True)
    
    _, best_path, best_cost, _ = find_best_path(start_loc, goal_loc, env, [W])
    feas_cost = env.path_cost(start_loc, path_to_moves(feas_path), W)[0]
    traj_cost = env.path_cost(start_loc, path_to_moves(traj), W)[0]
    
    best_edge_costs = env.path_cost(start_loc, path_to_moves(best_path), W)[1]
    feas_edge_costs = env.path_cost(start_loc, path_to_moves(feas_path), W)[1]
    traj_edge_costs = env.path_cost(start_loc, path_to_moves(traj), W)[1]
    
    vis_path(best_path, ax, label=f'Best: {best_cost:.2f}')
    vis_path(feas_path, ax, color='green', label=f'Feasible: {feas_cost:.2f}')
    vis_path(traj, ax, label=f'Actual {traj_cost:.2f}')
    
    # Also plot the edge costs
    ax.scatter(best_path[:-1,0]+0.5, best_path[:-1,1]+0.5, marker='o',
                c=get_colors(best_edge_costs), s=50)
    set_colorbar(best_edge_costs, ax)
    
    ax.scatter(feas_path[:-1,0]+0.5, feas_path[:-1,1]+0.5, marker='o', 
                c=get_colors(feas_edge_costs, vmin=np.min(best_edge_costs),
                            vmax=np.max(best_edge_costs)), s=50)
    
    ax.scatter(traj[:-1,0]+0.5, traj[:-1,1]+0.5, marker='o', 
                c=get_colors(traj_edge_costs, vmin=np.min(best_edge_costs),
                            vmax=np.max(best_edge_costs)), s=50)
        
    plt.legend(loc='best')
    plot_wind_vector(W, ax, alpha=0.5, color='black')
    plt.title(r"$\gamma$ = " + str(gamma))
        
### Plot gamma code
def plot_gammas():
    gammas = np.linspace(0, 1, 10)
    sub_optimality = []
    n_samples = 5
    for g in gammas:
        val = 0
        for i in range(n_samples):
            start_loc = np.array([0, 0])
            goal_loc = np.array([8, 9])
            grid_size = (10, 15)
            n_obst = 5
            edge_size = 1
            
            
            feas_path, occ = create_feasible_occ(n_obst, grid_size, start_loc, goal_loc)
            env = Environment(occ, 1, 0.1, buildings_block=False)  
            W, dists, locs = create_wind_vector(grid_size, g) 
            num_disc = 50
            gamma_vals = np.linspace(0.01, 1, num_disc, endpoint=True)
            # Uniform initialization of gamma distribution i.e. uniform prior on gamma
            # Use log pdf
            prior_dist = np.log(1/num_disc * np.ones(num_disc))
            
            num_gamma = 10
            num_draws = 10
            # TODO: need two measurements for meaning update 
            # take measurement 
            curr_loc = start_loc
            traj = [start_loc]
            while (curr_loc != goal_loc).any():
                new_wind_meas = env.perceived_wind(W, curr_loc) # TODO: call raw wind and do without buildings
                new_dist_rec = update_gamma_dist(gamma_vals, prior_dist, np.expand_dims(new_wind_meas, axis=0), [curr_loc])
                gamma_draws, W_draws = sample_posterior_wind_vector(grid_size, gamma_vals, new_dist_rec, 
                                                 num_gamma, num_draws, np.expand_dims(new_wind_meas, axis=0), np.expand_dims(curr_loc, axis=0))
                W_list = W_draws.reshape((-1, grid_size[0], grid_size[1], 2))
                W_avg = np.mean(W_list, axis=0)
                feasible, path, avg_cost, path_costs = find_best_path(curr_loc, goal_loc, env, W_list)
                moves = path_to_moves(path)
                next_loc = curr_loc + moves[0]
                traj.append(next_loc)
                curr_loc = next_loc
                
            traj = np.array(traj)
            
            # fig, ax = plt.subplots()
            # vis_map(occ, ax)
            # plt.grid(True)
            
            _, best_path, best_cost, _ = find_best_path(start_loc, goal_loc, env, [W])
            # feas_cost = env.path_cost(start_loc, path_to_moves(feas_path), W)[0]
            traj_cost = env.path_cost(start_loc, path_to_moves(traj), W)[0]
            
            # best_edge_costs = env.path_cost(start_loc, path_to_moves(best_path), W)[1]
            # feas_edge_costs = env.path_cost(start_loc, path_to_moves(feas_path), W)[1]
            # traj_edge_costs = env.path_cost(start_loc, path_to_moves(traj), W)[1]
            
            # vis_path(best_path, ax, label=f'Best: {tot_cost:.2f}')
            # vis_path(feas_path, ax, color='green', label=f'Feasible: {feas_cost:.2f}')
            # vis_path(traj, ax, label=f'Actual {traj_cost:.2f}')
            
            # # Also plot the edge costs
            # ax.scatter(best_path[:-1,0]+0.5, best_path[:-1,1]+0.5, marker='o',
            #            c=get_colors(best_edge_costs), s=50)
            # set_colorbar(best_edge_costs, ax)
            
            # ax.scatter(feas_path[:-1,0]+0.5, feas_path[:-1,1]+0.5, marker='o', 
            #            c=get_colors(feas_edge_costs, vmin=np.min(best_edge_costs),
            #                         vmax=np.max(best_edge_costs)), s=50)
            
            # ax.scatter(traj[:-1,0]+0.5, traj[:-1,1]+0.5, marker='o', 
            #            c=get_colors(traj_edge_costs, vmin=np.min(best_edge_costs),
            #                         vmax=np.max(best_edge_costs)), s=50)
                
            # plt.legend(loc='best')
            val += (traj_cost - best_cost)/best_cost
            
        sub_optimality.append(val/n_samples)
    
    plt.plot(gammas, sub_optimality)
    plt.xlabel("gamma")
    plt.ylabel("relative increase in cost")