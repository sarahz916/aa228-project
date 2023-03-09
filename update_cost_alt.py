import numpy as np
import matplotlib.pyplot as plt
from bayesian_updating import update_gamma_dist, sample_posterior_wind_vector
from env import Environment, get_colors, set_colorbar
from env_gen import vis_path, vis_map, create_feasible_occ
from create_wind_vector import create_wind_vector, plot_wind_vector
from find_one_path import find_best_path, path_to_moves

def receding_horizon_plan(start_loc, goal_loc, W, gamma_vals, prior_dist, env, num_gamma, num_draws):   
    """Plan under currrent wind posterior, move, update posterior, repeat."""
    grid_size = env.occ.shape
    new_dist_rec = np.copy(prior_dist)

    # take measurement 
    curr_loc = np.copy(start_loc)
    traj = [start_loc]
    while (curr_loc != goal_loc).any():
        # 1. Collect the new measurement
        # When buildings block wind, we also only observe the unblocked part
        if env.buildings_block:
            new_wind_meas = env.perceived_wind(W, curr_loc)
        # If acting like buildings don't block, we should observe full wind
        else:
            new_wind_meas = W[curr_loc[0], curr_loc[1], :]
        
        # 2. Bayesian recursive posterior update
        new_dist_rec = update_gamma_dist(gamma_vals, new_dist_rec, 
                            np.expand_dims(new_wind_meas, axis=0), [curr_loc])
        
        # 3. Sample from the updated posterior
        gamma_draws, W_draws = sample_posterior_wind_vector(grid_size, gamma_vals, new_dist_rec, 
            num_gamma, num_draws, np.expand_dims(new_wind_meas, axis=0), np.expand_dims(curr_loc, axis=0))
        # Reformat so list of wind fields and compute average wind field
        W_list = W_draws.reshape((-1, grid_size[0], grid_size[1], 2))
        
        # 4. Compute best path under current posterior
        feasible, path, avg_cost, path_costs = find_best_path(curr_loc, goal_loc, env, W_list)
        
        # 5. Execute first step of found path
        moves = path_to_moves(path)
        next_loc = curr_loc + moves[0]
        traj.append(next_loc)
        curr_loc = next_loc
        
    traj = np.array(traj)
    
    return traj

def suboptimality_study(n_samples, gamma_range):
    """Characterize suboptimality as a function of gamma.""" 
    start_loc = np.array([0, 0])
    goal_loc = np.array([8, 9])
    grid_size = (10, 15)
    n_obst = 5
    
    num_disc = 50
    num_gamma = 10
    num_draws = 10
    
    # Uniform initialization of gamma distribution i.e. uniform prior on gamma
    # Use log pdf
    gamma_vals = np.linspace(0.01, 1, num_disc, endpoint=True)
    prior_dist = np.log(1/num_disc * np.ones(num_disc))
    
    feas_costs = np.zeros((len(gamma_range), n_samples))
    actual_costs =  np.zeros((len(gamma_range), n_samples))
    best_costs = np.zeros((len(gamma_range), n_samples))
    
    for i, g in enumerate(gamma_range):
        print('On gamma = ' + str(g))
        for j in range(n_samples):
            print('On sample ' + str(j))
            feas_path, occ = create_feasible_occ(n_obst, grid_size, start_loc, goal_loc)
            env = Environment(occ, 1, 0.1, buildings_block=False)  
            W, dists, locs = create_wind_vector(grid_size, g) 
            
            traj = receding_horizon_plan(start_loc, goal_loc, W, gamma_vals, 
                                         prior_dist, env, num_gamma, num_draws)
            
            _, best_path, best_cost, _ = find_best_path(start_loc, goal_loc, env, [W])
            feas_cost = env.path_cost(start_loc, path_to_moves(feas_path), W)[0]
            traj_cost = env.path_cost(start_loc, path_to_moves(traj), W)[0]
            
            feas_costs[i,j] = feas_cost
            actual_costs[i,j] = traj_cost
            best_costs[i,j] = best_cost
    
    fig = plt.figure()

    plt.plot(gamma_range, np.mean(feas_costs, axis=1), label='Feasible')
    plt.plot(gamma_range, np.mean(actual_costs, axis=1), label='Actual')
    plt.plot(gamma_range, np.mean(best_costs, axis=1), label='Best')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Average Trajectory Cost')
    plt.title(r'Comparing Trajectory Costs Across $\gamma$')
    plt.legend()
    
    return feas_costs, actual_costs, best_costs, fig

if __name__ == '__main__':    
    start_loc = np.array([0, 0])
    goal_loc = np.array([8, 9])
    grid_size = (10, 15)
    n_obst = 5
    edge_size = 1
    gamma = 0.1
    
    feas_path, occ = create_feasible_occ(n_obst, grid_size, start_loc, goal_loc)
    env = Environment(occ, 1, 0.1, buildings_block=False)  
    
    num_disc = 50
    num_gamma = 10
    num_draws = 10
    W, dists, locs = create_wind_vector(grid_size, gamma) 
    gamma_vals = np.linspace(0.01, 1, num_disc, endpoint=True)
    # Uniform initialization of gamma distribution i.e. uniform prior on gamma
    # Use log pdf
    prior_dist = np.log(1/num_disc * np.ones(num_disc))
        
    traj = receding_horizon_plan(start_loc, goal_loc, W, gamma_vals, 
                                 prior_dist, env, num_gamma, num_draws)
    
    fig, ax = plt.subplots()
    vis_map(occ, ax)
    
    _, best_path, _, _ = find_best_path(start_loc, goal_loc, env, [W])
    best_cost, best_edge_costs = env.path_cost(start_loc, path_to_moves(best_path), W)
    feas_cost, feas_edge_costs = env.path_cost(start_loc, path_to_moves(feas_path), W)
    traj_cost, traj_edge_costs = env.path_cost(start_loc, path_to_moves(traj), W)
    
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
    
    # Overlay the true wind field
    plot_wind_vector(W, ax, alpha=0.5, color='black')
    plt.title(r"$\gamma = $" + str(gamma))
    
    n_samples = 200
    gamma_vals = np.linspace(0.01, 0.5, 10)
    feas_costs, actual_costs, best_costs, fig = suboptimality_study(n_samples, gamma_vals)
    