import numpy as np 
import matplotlib.pyplot as plt

plt.close('all')

def create_wind_vector(grid_size, gamma):
    W = np.zeros((grid_size[0], grid_size[1], 2))
    
    # List all cells by index
    locs = np.transpose(np.where(W[:,:,0] == 0))
        
    # reshaped = np.reshape(np.arange(len(locs)), grid_size)
    # print('locs', locs)
    # print('reshaped', reshaped)
    
    # Compute the distance between all cells (ignoring obstacles)
    dists = np.zeros((len(locs), len(locs)))
    for i, x1 in enumerate(locs):
        for j, x2 in enumerate(locs):
            dists[i,j] = np.sqrt(np.sum(np.square(x1 - x2))) # np.sum(np.abs(x1 - x2))
            # print('x1', x1, 'x2', x2, 'dists[i,j]', dists[i,j])        
    cov = np.exp(-gamma * np.square(dists))
     
    for k in range(2):
        flat_wind = np.random.multivariate_normal(np.zeros(len(dists)), cov)
        # Reshape into grid format
        W[:,:,k] = np.reshape(flat_wind, grid_size)
        
    return W, dists, locs

def plot_wind_vector(W, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    ind_0 = np.arange(0, W.shape[0])
    ind_1 = np.arange(0, W.shape[1]) 
    loc_0, loc_1 = np.meshgrid(ind_0, ind_1)

    ax.quiver(loc_0, loc_1, W[:,:,0], W[:,:,1], **kwargs)
    
    return ax

if __name__ == '__main__':
    grid_size = (10, 10)
    W, dists, locs = create_wind_vector(grid_size, 0.1) 
    plot_wind_vector(W, None, alpha=0.5, color='grey')
    