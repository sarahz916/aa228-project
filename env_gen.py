import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def find_feasible_path(start_ind, goal_ind, occupancy):
    """Check if a given occupancy map has a feasible solution."""
    # First create the adjacency graph, each cell is one state
    size = np.shape(occupancy)
    N = np.prod(size)
    
    # Construct the adjacency graph
    A = np.zeros((N,N))
    for i in range(occupancy.shape[0]):
        for j in range(occupancy.shape[1]):
            if occupancy[i,j] == 1: continue
            myInd = np.ravel_multi_index((i,j),size)
            if j > 0 and occupancy[i,j-1] == 0:
                neighbor = np.ravel_multi_index((i,j-1),size)
                A[neighbor, myInd] = 1
            if j < occupancy.shape[1]-1 and occupancy[i,j+1] == 0:
                neighbor = np.ravel_multi_index((i,j+1),size)
                A[neighbor, myInd] = 1
            if i > 0 and occupancy[i-1,j] == 0:
                neighbor = np.ravel_multi_index((i-1,j),size)
                A[neighbor, myInd] = 1
            if i < occupancy.shape[0]-1 and occupancy[i+1,j] == 0:
                neighbor = np.ravel_multi_index((i+1,j),size)
                A[neighbor, myInd] = 1
    A_sparse = csr_matrix(A)
        
    start = np.ravel_multi_index(start_ind,size)
    goal = np.ravel_multi_index(goal_ind,size)
    
    dist_matrix, predecessors = shortest_path(A_sparse, directed=False,
                                              unweighted=True, 
                                              return_predecessors=True, 
                                              indices=start)
    feasible = (dist_matrix[goal] != np.inf)
    
    path = []
    
    if feasible:
        curr = goal
        while curr != -9999:
            path.append(np.unravel_index(curr,size))
            curr = predecessors[curr]
    
    path = path[::-1]
    path = np.array(path)
    
    return feasible, path
    
def create_feasible_env(n_obst, grid_size, start_ind, goal_ind):
    """Creates an environment which is guaranteed to be feasible."""        
    disp = goal_ind - start_ind
    
    # Check that can actually place that many obstacles
    minPathLength = np.abs(disp[0]) + np.abs(disp[1]) + 1
    remain = grid_size[0] * grid_size[1] - n_obst - minPathLength
    
    if remain <= 0:
        raise Exception('Cannot place that many obstacles')
    
    while True:
        occupancy = np.zeros(grid_size)
    
        # Now, randomly place n_obst obstacles in occupancy grid
        allInds = np.array(list(range(grid_size[0] * grid_size[1])))
        chosenInds = np.random.choice(allInds, size=n_obst, replace=False)
        chosenInds = np.unravel_index(chosenInds, grid_size)
        occupancy[chosenInds] = 1
        
        # Check whether can connect start_ind and goal_ind using BFS
        feasible, path = find_feasible_path(start_ind, goal_ind, occupancy)
        
        if feasible:
            break
    
    return path, occupancy
    
def vis_path(path_inds, ax=None, **kwargs):
    path = np.array(path_inds, dtype='float')
    # Offset so in center
    path[:,0] += 0.5
    path[:,1] += 0.5
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(path[:,0], path[:,1], **kwargs)

    return ax

def vis_map(occupancy, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    obstLocs = np.transpose(np.where(occupancy==1))
        
    for loc in obstLocs:
        vertices = np.array([loc, 
                    [loc[0], loc[1]+1],
                    [loc[0]+1, loc[1]+1],
                    [loc[0]+1, loc[1]]])
        p = Polygon(vertices, alpha=0.5, 
                    facecolor='k', edgecolor = 'k')
        ax.add_patch(p)
        ax.set_xlim(0, occupancy.shape[0])
        ax.set_ylim(0, occupancy.shape[1])

    return ax

if __name__ == '__main__':
    plt.close('all')
    
    start_ind = np.array([0, 0])
    goal_ind = np.array([8, 9])
    grid_size = (10, 15)
    n_obst = 50
    edge_size = 1
    
    path, occupancy = create_feasible_env(n_obst, grid_size, start_ind, goal_ind)
    
    # Visualize the environment and the path
    fig, ax = plt.subplots()
    vis_map(occupancy, ax)
    vis_path(path, ax)
    plt.grid(True)