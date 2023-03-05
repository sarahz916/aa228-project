import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from env_gen import vis_path, vis_map, create_feasible_occ
from create_wind_vector import create_wind_vector, plot_wind_vector
from env import Environment, get_colors, set_colorbar

def path_to_moves(path):
    moves = np.array(path)
    moves = moves[1:] - moves[:-1]
    return moves

def build_cost_mat(env, all_wind_list):
    # First create the adjacency graph, each cell is one state
    occ = env.occ
    size = np.shape(occ)
    N = np.prod(size)
    
    # Construct the adjacency graph
    A = np.zeros((N,N))
    for i in range(occ.shape[0]):
        for j in range(occ.shape[1]):
            if occ[i,j] == 1: continue
        
            curr_loc = (i,j)
            myInd = np.ravel_multi_index(curr_loc, size)
                        
            if j > 0 and occ[i,j-1] == 0:
                move = [0, -1]
                neighbor = np.ravel_multi_index((i,j-1), size)
                A[myInd, neighbor] = env.get_avg_cost(curr_loc, move, all_wind_list)[0]         
                
            if j < occ.shape[1]-1 and occ[i,j+1] == 0:
                move = [0, 1]
                neighbor = np.ravel_multi_index((i,j+1), size)
                A[myInd, neighbor] = env.get_avg_cost(curr_loc, move, all_wind_list)[0]
            
            if i > 0 and occ[i-1,j] == 0:
                move = [-1, 0]
                neighbor = np.ravel_multi_index((i-1,j), size)
                A[myInd, neighbor] = env.get_avg_cost(curr_loc, move, all_wind_list)[0]
            
            if i < occ.shape[0]-1 and occ[i+1,j] == 0:
                move = [1, 0]
                neighbor = np.ravel_multi_index((i+1,j), size)
                A[myInd, neighbor] = env.get_avg_cost(curr_loc, move, all_wind_list)[0]
                
    A_sparse = csr_matrix(A)
    
    return A_sparse, A

def find_best_path(start_loc, goal_loc, env, all_wind_list):
    occ = env.occ
    size = np.shape(occ)
    
    A_sparse, A = build_cost_mat(env, all_wind_list)

    start = np.ravel_multi_index(start_loc, size)
    goal = np.ravel_multi_index(goal_loc, size)
    
    dist_matrix, predecessors = shortest_path(A_sparse, directed=True,
                                              unweighted=False, 
                                              return_predecessors=True, 
                                              indices=start)
    feasible = (dist_matrix[goal] != np.inf)
    
    path = []
    path_costs = np.inf
    avg_cost = np.inf
    
    if feasible:
        curr = goal
        while curr != -9999:
            path.append(np.unravel_index(curr, size))
            curr = predecessors[curr]
    
        path = np.array(path[::-1])
        moves = path_to_moves(path)
        
        # Compute the corresponding total path costs and average cost
        path_costs = [env.path_cost(start_loc, moves, all_wind)[0] for all_wind 
                      in all_wind_list]
        avg_cost = np.mean(path_costs)
        
    return feasible, path, avg_cost, path_costs

if __name__ == '__main__':
    plt.close('all')
    
    start_loc = np.array([0, 0])
    goal_loc = np.array([8, 9])
    grid_size = (10, 15)
    n_obst = 5
    edge_size = 1

    feas_path, occ = create_feasible_occ(n_obst, grid_size, start_loc, goal_loc)
    env = Environment(occ, 1, 0.1, buildings_block=True)
    
    # Visualize the environment and the path
    fig, ax = plt.subplots()
    vis_map(occ, ax)
    plt.grid(True)
    
    W, dists, locs = create_wind_vector(grid_size, 0.1) 
    plot_wind_vector(W, ax, alpha=0.5, color='black')
    
    feas_cost = env.path_cost(start_loc, path_to_moves(feas_path), W)[0]
    
    _, best_path, tot_cost, _ = find_best_path(start_loc, goal_loc, env, [W])
    
    best_edge_costs = env.path_cost(start_loc, path_to_moves(best_path), W)[1]
    feas_edge_costs = env.path_cost(start_loc, path_to_moves(feas_path), W)[1]

    vis_path(best_path, ax, label=f'Best: {tot_cost:.2f}')
    vis_path(feas_path, ax, color='green', label=f'Feasible: {feas_cost:.2f}')

    # Also plot the edge costs
    ax.scatter(best_path[:-1,0]+0.5, best_path[:-1,1]+0.5, marker='o',
               c=get_colors(best_edge_costs), s=50)
    set_colorbar(best_edge_costs, ax)
    
    ax.scatter(feas_path[:-1,0]+0.5, feas_path[:-1,1]+0.5, marker='o', 
               c=get_colors(feas_edge_costs, vmin=np.min(best_edge_costs),
                            vmax=np.max(best_edge_costs)), s=50)
        
    plt.legend(loc='best')

    
    
    