import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from env_gen import vis_path, vis_map, create_feasible_env
from create_wind_vector import create_wind_vector, plot_wind_vector

def get_colors(inp, colormap=plt.cm.viridis, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(inp)
    if vmax is None:
        vmax = np.max(inp)
    
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def set_colorbar(arr, ax=None, cmap='viridis', vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(arr)
    if vmax is None:
        vmax = np.max(arr)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    if ax is None:
        plt.colorbar(sm)
    else:
        plt.colorbar(sm, ax=ax)

# TODO: TEMPORARY 
def one_step_cost(curr_loc, move, all_wind, occ):
    p_w = all_wind[tuple(curr_loc)]
    cost = max(0.1, 1-np.inner(p_w, move))
    # cost = 1 
    return cost

def get_edge_costs(start_loc, moves, all_wind, occ):
    edge_costs = []
    curr_loc = np.copy(start_loc)
    for m in moves:
        edge_costs.append(one_step_cost(curr_loc, m, all_wind, occ))
        curr_loc += m
    return edge_costs
    
# TODO: TEMPORARY 
def path_cost(start_loc, moves, all_wind, occ):
    total_cost = 0
    curr_loc = np.copy(start_loc)
    for m in moves:
        total_cost += one_step_cost(curr_loc, m, all_wind, occ)
        curr_loc += m
    return total_cost

def get_avg_cost(curr_loc, move, all_wind_list, occ):
    costs = [one_step_cost(curr_loc, move, all_wind, occ) for
             all_wind in all_wind_list]
    return np.mean(costs), costs

def path_to_moves(path):
    moves = np.array(path)
    moves = moves[1:] - moves[:-1]
    return moves

def build_cost_mat(occ, all_wind_list):
    # First create the adjacency graph, each cell is one state
    size = np.shape(occ)
    N = np.prod(size)
    
    # Construct the adjacency graph
    A = np.zeros((N,N))
    for i in range(occ.shape[0]):
        for j in range(occ.shape[1]):
            if occ[i,j] == 1: continue
        
            curr_loc = (i,j)
            myInd = np.ravel_multi_index(curr_loc, size)
                        
            # Consider moving up if feasible
            if j > 0 and occ[i,j-1] == 0:
                move = [0, -1]
                neighbor = np.ravel_multi_index((i,j-1), size)
                A[myInd, neighbor] = get_avg_cost(curr_loc, move, all_wind_list, occ)[0]         
                
            # Consider moving down if feasible
            if j < occ.shape[1]-1 and occ[i,j+1] == 0:
                move = [0, 1]
                neighbor = np.ravel_multi_index((i,j+1),size)
                A[myInd, neighbor] = get_avg_cost(curr_loc, move, all_wind_list, occ)[0]
            
            # Consider moving left if feasible
            if i > 0 and occ[i-1,j] == 0:
                move = [-1, 0]
                neighbor = np.ravel_multi_index((i-1,j),size)
                A[myInd, neighbor] = get_avg_cost(curr_loc, move, all_wind_list, occ)[0]
            
            # Consider moving right if feasible
            if i < occ.shape[0]-1 and occ[i+1,j] == 0:
                move = [1, 0]
                neighbor = np.ravel_multi_index((i+1,j), size)
                A[myInd, neighbor] = get_avg_cost(curr_loc, move, all_wind_list, occ)[0]
                
    A_sparse = csr_matrix(A)
    
    return A_sparse, A

def find_best_path(start_ind, goal_ind, occ, all_wind_list):
    size = np.shape(occ)
    
    A_sparse, A = build_cost_mat(occ, all_wind_list)

    start = np.ravel_multi_index(start_ind, size)
    goal = np.ravel_multi_index(goal_ind, size)
    
    dist_matrix, predecessors = shortest_path(A_sparse, directed=True,
                                              unweighted=False, 
                                              return_predecessors=True, 
                                              indices=start)
    feasible = (dist_matrix[goal] != np.inf)
    
    path = []
    path_costs = np.inf
    tot_cost = np.inf
    
    if feasible:
        curr = goal
        while curr != -9999:
            path.append(np.unravel_index(curr, size))
            curr = predecessors[curr]
    
        path = np.array(path[::-1])
        moves = path_to_moves(path)
        
        # Compute the corresponding total path costs and average cost
        path_costs = [path_cost(start_ind, moves, all_wind, occ) for all_wind 
                      in all_wind_list]
        tot_cost = np.mean(path_costs)
        
    return feasible, path, tot_cost, path_costs

# def vis_cost_mat(A, size, ax=None, **kwargs):
#     if ax is None:
#         fig, ax = plt.subplots()
        
#     arrows = []
#     costs = []
    
#     for i in range(size[0]):
#         for j in range(size[1]):        
#             curr_loc = (i,j)
#             myInd = np.ravel_multi_index(curr_loc, size)
             
#             # Consider moving up
#             if j > 0:
#                 neighbor = np.ravel_multi_index((i,j-1), size)
#                 if A[neighbor, myInd] > 0:
#                     arrows.append([i+0.5, j+0.5, 0, -1/2])
#                     costs.append(A[neighbor, myInd])
                    
#             # Consider moving down
#             if j < size[1]-1:
#                 neighbor = np.ravel_multi_index((i,j+1), size)
#                 if A[neighbor, myInd] > 0:
#                     arrows.append([i+0.5, j+0.5, 0, 1/2])
#                     costs.append(A[neighbor, myInd])
                                
#             # Consider moving left
#             if i > 0:
#                 neighbor = np.ravel_multi_index((i-1,j), size)
#                 if A[neighbor, myInd] > 0:
#                     arrows.append([i+0.5, j+0.5, -1/2, 0])
#                     costs.append(A[neighbor, myInd])
                                
#             # Consider moving right
#             if i < size[0]-1:
#                 neighbor = np.ravel_multi_index((i+1,j), size)
#                 if A[neighbor, myInd] > 0:
#                     arrows.append([i+0.5, j+0.5, 1/2, 0])
#                     costs.append(A[neighbor, myInd])
        
#         colors = get_colors(costs)
        
#     for k, arrow in enumerate(arrows):
#         x, y, dx, dy = arrow
#         ax.arrow(x, y, dx, dy, color=colors[k])
    
#     set_colorbar(colors, ax)    
        
#     return ax

if __name__ == '__main__':
    plt.close('all')
    
    # Aaron TODO: Be consistent with Sarah
    # Sarah: (# rows, # cols) and (0, 0) is bottom left
    
    start_ind = np.array([0, 0])
    goal_ind = np.array([8, 9])
    grid_size = (15, 10)
    n_obst = 5
    edge_size = 1
    
    feas_path, occ = create_feasible_env(n_obst, grid_size, start_ind, goal_ind)
    
    # Visualize the environment and the path
    fig, ax = plt.subplots()
    vis_map(occ, ax)
    plt.grid(True)
    
    W, dists, locs = create_wind_vector(grid_size, 0.01) 
    plot_wind_vector(W, ax, alpha=0.5, color='black')
    
    feas_cost = path_cost(start_ind, path_to_moves(feas_path), W, occ)
    
    feasible, best_path, tot_cost, path_costs = find_best_path(start_ind, 
                                                        goal_ind, occ, [W])
        
    best_edge_costs = get_edge_costs(start_ind, path_to_moves(best_path), W, occ)
    feas_edge_costs = get_edge_costs(start_ind, path_to_moves(feas_path), W, occ)

    vis_path(best_path, ax, label='Best: ' + str(tot_cost))
    vis_path(feas_path, ax, color='green', label='Feasible: ' + str(feas_cost))

    # Also plot the edge costs
    ax.scatter(best_path[:-1,0]+0.5, best_path[:-1,1]+0.5, marker='o', 
               c=get_colors(best_edge_costs))
    set_colorbar(best_edge_costs, ax)
    
    ax.scatter(feas_path[:-1,0]+0.5, feas_path[:-1,1]+0.5, marker='o', 
               c=get_colors(feas_edge_costs, vmin=np.min(best_edge_costs), 
                            vmax=np.max(best_edge_costs)))
        
    plt.legend()

    
    
    