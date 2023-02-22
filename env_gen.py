import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def find_feasible_path(start_loc, goal_loc, occ):
    """Check if a given occ map has a feasible solution."""
    # First create the adjacency graph, each cell is one state
    size = np.shape(occ)
    N = np.prod(size)
    
    # Construct the adjacency graph
    A = np.zeros((N,N))
    for i in range(occ.shape[0]):
        for j in range(occ.shape[1]):
            if occ[i,j] == 1: continue
            myInd = np.ravel_multi_index((i,j),size)
            if j > 0 and occ[i,j-1] == 0:
                neighbor = np.ravel_multi_index((i,j-1),size)
                A[neighbor, myInd] = 1
            if j < occ.shape[1]-1 and occ[i,j+1] == 0:
                neighbor = np.ravel_multi_index((i,j+1),size)
                A[neighbor, myInd] = 1
            if i > 0 and occ[i-1,j] == 0:
                neighbor = np.ravel_multi_index((i-1,j),size)
                A[neighbor, myInd] = 1
            if i < occ.shape[0]-1 and occ[i+1,j] == 0:
                neighbor = np.ravel_multi_index((i+1,j),size)
                A[neighbor, myInd] = 1
    A_sparse = csr_matrix(A)
        
    start = np.ravel_multi_index(start_loc,size)
    goal = np.ravel_multi_index(goal_loc,size)
    
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
    
def create_feasible_occ(n_obst, grid_size, start_loc, goal_loc):
    """Creates an occupancy which is guaranteed to be feasible."""        
    disp = goal_loc - start_loc
    
    # Check that can actually place that many obstacles
    minPathLength = np.abs(disp[0]) + np.abs(disp[1]) + 1
    remain = grid_size[0] * grid_size[1] - n_obst - minPathLength
    
    if remain <= 0:
        raise Exception('Cannot place that many obstacles')
    
    while True:
        occ = np.zeros(grid_size)
    
        # Now, randomly place n_obst obstacles in occ grid
        allInds = np.array(list(range(grid_size[0] * grid_size[1])))
        chosenInds = np.random.choice(allInds, size=n_obst, replace=False)
        chosenInds = np.unravel_index(chosenInds, grid_size)
        occ[chosenInds] = 1
        
        # Check whether can connect start_loc and goal_loc using BFS
        feasible, path = find_feasible_path(start_loc, goal_loc, occ)
        
        if feasible:
            break
    
    return path, occ
    
def vis_path(path_inds, ax=None, **kwargs):
    path = np.array(path_inds, dtype='float')
    # Offset so in center
    path[:,0] += 0.5
    path[:,1] += 0.5
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(path[:,0], path[:,1], **kwargs)

    return ax

def vis_map(occ, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    obstLocs = np.transpose(np.where(occ==1))
        
    for loc in obstLocs:
        vertices = np.array([loc, 
                    [loc[0], loc[1]+1],
                    [loc[0]+1, loc[1]+1],
                    [loc[0]+1, loc[1]]])
        p = Polygon(vertices, alpha=0.5, 
                    facecolor='k', edgecolor = 'k')
        ax.add_patch(p)
        ax.set_xlim(0, occ.shape[0])
        ax.set_ylim(0, occ.shape[1])

    return ax

if __name__ == '__main__':
    plt.close('all')
    
    start_loc = np.array([0, 0])
    goal_loc = np.array([8, 9])
    grid_size = (10, 15)
    n_obst = 50
    edge_size = 1
    
    path, occ = create_feasible_occ(n_obst, grid_size, start_loc, goal_loc)
    
    # Visualize the environment and the path
    fig, ax = plt.subplots()
    vis_map(occ, ax)
    vis_path(path, ax)
    plt.grid(True)