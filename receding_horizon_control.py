import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from bayesian_updating import update_gamma_dist_1d, sample_posterior_wind_vector_1d, belief_to_dist
from env import Environment, get_colors, set_colorbar
from env_gen import vis_path, vis_map, create_feasible_occ
from create_wind_vector import create_wind_vector, plot_wind_vector_and_speeds, plot_wind_vector
from find_one_path import find_best_path, path_to_moves

def receding_horizon_plan(start_loc, goal_loc, W, gamma_vals, prior_dist, env, 
                          num_gamma, num_draws, verbose=False):   
    """Plan under currrent wind posterior, move, update posterior, repeat."""
    grid_size = env.occ.shape
    new_dist_rec = np.copy(prior_dist)

    # Will iteratively grow these arrays as move in the environment
    old_locs_x = []
    old_locs_y = []
    old_wind_meas_x = []
    old_wind_meas_y = []
    sign_locs_x = []
    sign_locs_y = []
    wind_signs_x = []
    wind_signs_y = []
    
    # Also track some items for visualization/debugging
    W_list = [] # All the wind draws we take organized as list of arrays
    gamma_beliefs = [] # Store intermediate beliefs over gamma
    
    # Add starting point
    curr_loc = np.copy(start_loc)
    traj = [curr_loc]
    
    # Terminate once reach the goal
    while (curr_loc != goal_loc).any():
        if verbose:
            print('Current Location: ', curr_loc)
        
        # 1. Collect the new measurement
        # When buildings block wind, we also only observe the unblocked part
        if env.buildings_block:
            new_wind_meas_x, sign_x = env.perceived_wind_x(W, curr_loc)
            new_wind_meas_y, sign_y = env.perceived_wind_y(W, curr_loc)
        # If acting like buildings don't block, we should observe full wind
        else:
            new_wind_meas = W[curr_loc[0], curr_loc[1], :]
            new_wind_meas_x, new_wind_meas_y = new_wind_meas
            sign_x = 0
            sign_y = 0
        
        # 2. Bayesian recursive posterior update
        
        # If we have already visited this location it provides no new info
        # so skip the update
        # Note: Adding duplicate measurements will also break things
        if np.any([(curr_loc == prev_loc).all() for prev_loc in traj[:-1]]):
            new_dist_rec = np.copy(new_dist_rec)
        else:
            # First do for the x component
            if new_wind_meas_x != None:
                new_dist_rec = update_gamma_dist_1d(gamma_vals, new_dist_rec, 
                                                 np.expand_dims(new_wind_meas_x, axis=0), [curr_loc],
                                                 old_wind_meas_x, old_locs_x) 
                old_locs_x.append(curr_loc)
                old_wind_meas_x.append(new_wind_meas_x)
            elif sign_x != 0:
                sign_locs_x.append(curr_loc)
                wind_signs_x.append(sign_x)
            # Then do for the y component
            if new_wind_meas_y != None:
                new_dist_rec = update_gamma_dist_1d(gamma_vals, new_dist_rec, 
                                                 np.expand_dims(new_wind_meas_y, axis=0), [curr_loc],
                                                 old_wind_meas_y, old_locs_y) 
                old_locs_y.append(curr_loc)
                old_wind_meas_y.append(new_wind_meas_y)
            elif sign_y != 0:
                sign_locs_y.append(curr_loc)
                wind_signs_y.append(sign_y)
        
        gamma_beliefs.append(new_dist_rec)
        
        # 3. Sample from the updated posterior
        # First for the x
        gamma_draws, W_draws_x = sample_posterior_wind_vector_1d(grid_size, gamma_vals, new_dist_rec, 
                                         num_gamma, num_draws, np.array(old_wind_meas_x), 
                                         np.array(old_locs_x), np.array(wind_signs_x), np.array(sign_locs_x))
        # Reuse the same gamma draws for the y
        _, W_draws_y = sample_posterior_wind_vector_1d(grid_size, gamma_vals, new_dist_rec, 
                                         num_gamma, num_draws, np.array(old_wind_meas_y), 
                                         np.array(old_locs_y), np.array(wind_signs_y), np.array(sign_locs_y),
                                         gamma_draws = gamma_draws)
        # Concatenate to form the overall wind vector fields
        W_draws = np.concatenate((W_draws_x, W_draws_y), axis = 4)
        # Don't care about distinction about what draws for which gamma value
        W_draws = W_draws.reshape((-1, grid_size[0], grid_size[1], 2))
        
        W_list.append(W_draws)
        
        # 4. Compute best path under current posterior
        feasible, path, avg_cost, path_costs = find_best_path(curr_loc, goal_loc, env, W_draws)
        moves = path_to_moves(path)
        
        if verbose:
            print('Found feasible path?', feasible)
        
        # 5. Execute first step of found path
        next_loc = curr_loc + moves[0]
        traj.append(next_loc)
        curr_loc = next_loc
    
    traj = np.array(traj)

    return traj, W_list, gamma_beliefs

if __name__ == '__main__':    
    start_loc = np.array([0, 0])
    goal_loc = np.array([8, 9])
    grid_size = (10, 15)
    n_obst = 30
    edge_size = 1
    gamma = 0.1
    
    feas_path, occ = create_feasible_occ(n_obst, grid_size, start_loc, goal_loc)
    env = Environment(occ, 1, 0.1, buildings_block=True)  
    
    num_disc = 50
    num_gamma = 10
    num_draws = 10
    W, dists, locs = create_wind_vector(grid_size, gamma) 
    gamma_vals = np.linspace(0.01, 0.5, num_disc, endpoint=True)
    # Uniform initialization of gamma distribution i.e. uniform prior on gamma
    # Use log pdf
    prior_dist = np.log(1/num_disc * np.ones(num_disc))
            
    traj, W_list, gamma_beliefs = receding_horizon_plan(start_loc, goal_loc, W, 
                                gamma_vals, prior_dist, env, num_gamma, num_draws)
    iterations = range(1, len(W_list)+1)

    #### Visualize The Different Trajectories ####
    
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
    
    #########

    #### Visualize True v. Estimated Wind Field As Explore ####
        
    # First, get the mean estimated wind field at each time
    W_est_list = [np.mean(W_draws, axis=0) for W_draws in W_list]
    
    # Plot snapshots of the wind fields and true wind
    vis_times = [int(len(W_est_list)/4), int(len(W_est_list)/2), 
                 int(3*len(W_est_list)/4), len(W_est_list)-1] # when to take snapshots
    num_vis = len(vis_times)
    
    for i in range(num_vis+1):
        fig, ax = plt.subplots()
        
        if i != num_vis:
            vis_time = vis_times[i]
            plot_wind_vector_and_speeds(W_est_list[vis_time], ax)
                
            # Overlay the trajectory currently executed
            vis_path(traj[:vis_time]-0.5, ax, label='Executed Path', color='r')
            
            ax.set_title('Iteration ' + str(vis_time))
            ax.legend()

        else:
            plot_wind_vector_and_speeds(W, ax)
            ax.set_title('Truth')
        
    # Compare each of these to the true wind, look at RMSE
    rmse_list = [np.sqrt(np.mean(np.square(W - W_est))) for W_est in W_est_list]
    
    # Plot rmse over time
    plt.figure()
    plt.plot(iterations, rmse_list)
    plt.xticks(iterations)
    plt.xlabel('After Iteration')
    plt.ylabel('RMSE')
    plt.title('Estimated v. True Wind Field RMSE as Explore')
    
    #########

    #### Visualize Gamma Posterior As Explore ####
    plt.figure()
    for vis_time in vis_times:
        belief = gamma_beliefs[vis_time]
        gamma_dist = belief_to_dist(belief)
        plt.plot(gamma_vals, gamma_dist, label=vis_time)
    plt.title(r'Posterior over $\gamma$ at Varying Iterations')
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'Posterior Density')
    plt.legend()
    
    # Plot entropy as a function of time for gamma distribution
    entropies = [entropy(belief_to_dist(belief)) for belief in gamma_beliefs]
    # Normalize by the maximum entropy (uniform dist.)
    max_entropy = entropy(np.ones(len(gamma_beliefs[0])))
    entropies /= max_entropy
    plt.figure()
    plt.plot(iterations, entropies)
    plt.xlabel('After Iteration')
    plt.xticks(iterations)
    plt.ylabel('Normalized Entropy')
    plt.title(r'$\gamma$ Belief Entropy as Explore')
    
    ########








