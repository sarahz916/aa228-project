import numpy as np 
import matplotlib.pyplot as plt
from env import get_colors, set_colorbar

plt.close('all')

def create_wind_vector(grid_size, gamma):
    W = np.zeros((grid_size[0], grid_size[1], 2))
    
    # List all cells by index
    locs = np.transpose(np.where(W[:,:,0] == 0))
        
    # Compute the distance between all cells (ignoring obstacles)
    dists = np.zeros((len(locs), len(locs)))
    for i, x1 in enumerate(locs):
        for j, x2 in enumerate(locs):
            dists[i,j] = np.linalg.norm(x1 - x2)
    cov = np.exp(-gamma * np.square(dists))
     
    for k in range(2):
        flat_wind = np.random.multivariate_normal(np.zeros(len(dists)), cov)
        # Reshape into grid format
        W[:,:,k] = np.reshape(flat_wind, grid_size)
        
    return W, dists, locs

def plot_wind_vector(W, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    x_vals = np.arange(0, W.shape[0])
    y_vals = np.arange(0, W.shape[1]) 
    X, Y = np.meshgrid(x_vals, y_vals)
    X = X.T
    Y = Y.T
    # It should now be the case that X[i,j] = i, Y[i,j] = Y
    # Hence for given i,j:
    # X[i,j] = i, Y[i,j] = Y, W[i,j,0] = wx[i,j], W[i,j,1] = wy[i,j]
    
    ax.quiver(X, Y, W[:,:,0], W[:,:,1], angles='xy', scale_units='xy', scale=1, **kwargs)
    # Needed to ensure that the arrows actually render with correct aspect ratios
    ax.axis('equal')
    
    return ax

def plot_wind_vector_and_speeds(W, ax=None, **kwargs):
    # Can verify that the wind speeds plot is correct by looking at max speed
    # np.unravel_index(np.argmax(wind_speeds), wind_speeds.shape)    
    if ax is None:
        fig, ax = plt.subplots()
    
    wind_speeds = np.linalg.norm(W, axis=2)
    
    img = ax.imshow(wind_speeds.T) # transpose because imshow expects y first
    ax.invert_yaxis()
    ax.set_title('Wind Speeds')
    plt.colorbar(img, ax=ax)
    plot_wind_vector(W, ax, **kwargs)
    
    return ax, wind_speeds

if __name__ == '__main__':
    # n, m (# x vals, # y vals)
    grid_size = (10, 15)
    gamma = 0.1
    W, dists, locs = create_wind_vector(grid_size, gamma) 
    
    # View: Index into W, wind_speeds as W[x,y] with 0,0 being bottom left
    
    plot_wind_vector_and_speeds(W, None, alpha=0.5, color='black')
    plt.title(r'$\gamma$ = ' + str(gamma))