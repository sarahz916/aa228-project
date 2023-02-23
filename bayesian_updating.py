import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from create_wind_vector import create_wind_vector

def cond_gaussian(b, mu_a, mu_b, A, B, C):
    B_inv = np.linalg.inv(B)
    mu_a_given_b = mu_a + C @ B_inv @ (b - mu_b)
    sigma_a_given_b = A - C @ B_inv @ C.T
    return mu_a_given_b, sigma_a_given_b

# Inputs:
# gamma_vals = discretization points/value for the gamma distribution/posterior
# curr_dist = discrete array giving associated beliefs over gamma values in log-space
# new_wind_meas = array of shape Nx2 containing the new wind measurements (wx, wy)
# new_locs = array of shape Nx2 containing the location of these measurements (x, y)
# old_wind_meas = like new_wind_meas but for previously incorporated wind measurements
# old_locs = like new_locs but for previous wind measurements
# Outputs:
# new_dist = new gamma discretized beliefs in log-space
def update_gamma_dist(gamma_vals, curr_dist, new_wind_meas, new_locs, old_wind_meas=None, old_locs=None):    
    have_prev_meas = old_wind_meas is not None and old_locs is not None
    
    if have_prev_meas:
        locs = np.vstack([new_locs, old_locs])
    else:
        locs = np.copy(new_locs)
        
    # Compute the distance between cells (ignoring obstacles)
    dists = np.zeros((len(locs), len(locs)))
    for i, x1 in enumerate(locs):
        for j, x2 in enumerate(locs):
            dists[i,j] = np.linalg.norm(x1 - x2)
    
    new_dist = np.copy(curr_dist)    
    
    for i, gamma in enumerate(gamma_vals):
        cov = np.exp(-gamma * np.square(dists))

        # Assume the x and y wind directions are independent so can just
        # multiply the two corresponding x, y pdfs
        for j in range(2):    
            if have_prev_meas:
                # Find the conditional gaussian distribution p(* | b, gamma)
                a_len = len(new_wind_meas)
                b_len = len(old_wind_meas)
                mu_a = np.zeros(a_len)
                mu_b = np.zeros(b_len)
                A = cov[:a_len, :a_len] # upper left block
                B = cov[a_len:, a_len:] # lower right block
                C = cov[:a_len, a_len:] # upper right block
                b = old_wind_meas[:,j]
                mu, sigma = cond_gaussian(b, mu_a, mu_b, A, B, C)
                
                # For numerical reasons, clip the minimum value to 0
                sigma = np.clip(sigma, 0, np.inf)
                
                # Verify that recover overall cov
                # assert np.all(cov == np.vstack([np.hstack([A, C]), np.hstack([C.T, B])]))
                
                print('mu', mu)
                print('sigma', sigma)
                
            else:
                mu = np.zeros(len(new_wind_meas))
                sigma = cov
                
            a = new_wind_meas[:,j]
            # Evaluate p(* | b, gamma) at the given a, works if b is empty too
            # Work in log-space
            # Note: Have to allow_singular=True for larger matrices to make
            # this work because of numerical issues
            cond_prob = multivariate_normal(mu, sigma, allow_singular=True).logpdf(a)
                
            # Compute updated likelihood 
            # p(b_tot = (a, b) | gamma) = p(a | b, gamma) * p(b | gamma)
            # We assume p(b | gamma) is stored in curr_dist so just multiply
            # by the new factor. Add when working in log space
            new_dist[i] += cond_prob
    
    # Note: Since working with log-probabilities and belief no longer normalize
    # Normalize new_dist to get the posterior 
    # (implicitly assumes a uniform prior on gamma) 
    # p(gamma | b_tot) = p(b_tot | gamma) * p(gamma) / p(b) \propto p(b_tot | gamma)  
    # assuming p(gamma) uniform
    # So we just compute p(b_tot | gamma) above then normalize  
    # new_dist /= np.sum(new_dist)
    
    return new_dist

def belief_to_dist(belief):
    temp = np.exp(belief)
    return temp / np.sum(temp)

if __name__ == '__main__':
    # n, m (# x vals, # y vals)
    grid_size = (2, 2)
    gamma = 0.1
    W, dists, locs = create_wind_vector(grid_size, gamma) 
    
    num_disc = 100
    gamma_vals = np.linspace(0.01, 1, num_disc, endpoint=True)
    # Uniform initialization of gamma distribution i.e. uniform prior on gamma
    # Use log pdf
    gamma_dist = np.log(1/num_disc * np.ones(num_disc))
    
    # First, try single-shot posterior updating
    new_wind_meas = np.reshape(W, (-1,2))
    new_locs = locs
    new_dist = update_gamma_dist(gamma_vals, gamma_dist, new_wind_meas, new_locs)
    
    # Now, try recursive posterior updating
    new_dist_rec = np.copy(gamma_dist)
    old_locs = None
    old_wind_meas = None
    for i, wind_meas in enumerate(new_wind_meas):
        new_dist_rec = update_gamma_dist(gamma_vals, gamma_dist, 
            np.expand_dims(wind_meas, axis=0), np.expand_dims(new_locs[i], axis=0), 
            old_wind_meas, old_locs)
        if i > 0:
            old_locs = np.vstack([old_locs, new_locs[i]])
            old_wind_meas = np.vstack([old_wind_meas, wind_meas])
        else:
            old_locs = np.expand_dims(new_locs[i], axis=0)
            old_wind_meas = np.expand_dims(wind_meas, axis=0)
      
    # Can run this to verify that old_wind_meas, old_locs are set correctly at end
    # new_dist_rec = update_gamma_dist(gamma_vals, gamma_dist, old_wind_meas, old_locs)
            
    plt.figure()
    plt.plot(gamma_vals, belief_to_dist(new_dist), label='Single-Shot')
    plt.plot(gamma_vals, belief_to_dist(new_dist_rec), label='Recursive')
    plt.title(r'Posterior over $\gamma$')
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'Posterior Density')
    plt.legend()

