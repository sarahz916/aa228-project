import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from create_wind_vector import create_wind_vector, plot_wind_vector_and_speeds

def roundPSD(A):
    """Round A nearly PSD to a PSD matrix for numerical stability."""
    w, v = np.linalg.eigh(A)
    
    rounded_L = np.clip(w, 0, np.inf)
    rounded_L = np.diag(rounded_L)
    
    return v @ rounded_L  @ np.linalg.inv(v)

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
# QUESITON: what is the difference between new and old measurements?
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
                
            else:
                mu = np.zeros(len(new_wind_meas))
                sigma = cov
                    
                # For numerical reasons, clip the minimum value to 0
                # sigma = np.clip(sigma, 0, np.inf)
                
            # For numerical reasons, round to nearest PSD matrix
            sigma = roundPSD(sigma)
                
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
    """Convert from a log-space belief to a normalized probability distribution."""
    temp = np.exp(belief)
    den = np.sum(temp)
    
    prob = temp / den
    
    # If belief becomes extremely concentrated around one value just return
    # dirac distribution because of numerical precision
    if np.any(np.isnan(prob)) or np.any(np.isinf(prob)):
        print('hi')
        temp = np.zeros(len(belief))
        temp[np.argmax(belief)] = 1
        return temp
    
    return prob

def get_hidden_locs(grid_size, meas_locs):
    """Find what locations are not measured in a grid."""
    W = np.zeros((grid_size[0], grid_size[1], 2))
    all_locs = np.transpose(np.where(W[:,:,0] == 0))
    x = set(list(map(tuple, all_locs))) - set(list(map(tuple, meas_locs)))
    hidden_locs = np.array(list(x))
    return hidden_locs

def sample_posterior_wind_vector(grid_size, gamma_vals, gamma_belief, 
                                 num_gamma, num_draws, wind_meas=None, meas_locs=None):
    """Given wind measurements and a gamma distribution, sample wind fields from the posterior."""
    have_prev_meas = wind_meas is not None and meas_locs is not None
        
    # List all cells by index
    hidden_locs = get_hidden_locs(grid_size, meas_locs)
    
    # Draw gamma samples from gamma_dist
    gamma_draws = np.random.choice(gamma_vals, num_gamma, p=belief_to_dist(gamma_belief), replace=True)
    
    if not len(hidden_locs):
        print("Posterior sampling is silly. All locations are measured.")
        W_draws = np.zeros((num_gamma, num_draws, grid_size[0], grid_size[1], 2))
        
        # TODO: Do in a single-step
        for k in range(len(meas_locs)):
            W_draws[:, :, meas_locs[k,0], meas_locs[k,1], :] = wind_meas[k,:]
                
        return gamma_draws, W_draws
    
    if have_prev_meas:
        locs = np.vstack([hidden_locs, meas_locs])
    else:
        locs = hidden_locs
    
    # Compute the distance between all cells (ignoring obstacles)
    dists = np.zeros((len(locs), len(locs)))
    for i, x1 in enumerate(locs):
        for j, x2 in enumerate(locs):
            dists[i,j] = np.linalg.norm(x1 - x2)
            
    W_draws = np.zeros((num_gamma, num_draws, grid_size[0], grid_size[1], 2))
    
    for i, gamma in enumerate(gamma_draws):
        cov = np.exp(-gamma * np.square(dists))

        # Assume the x and y wind directions are independent so can just
        # multiply the two corresponding x, y pdfs
        for j in range(2):    
            if have_prev_meas:
                # Find the conditional gaussian distribution p(* | b, gamma)
                a_len = len(hidden_locs)
                b_len = len(meas_locs)
                mu_a = np.zeros(a_len)
                mu_b = np.zeros(b_len)
                A = cov[:a_len, :a_len] # upper left block
                B = cov[a_len:, a_len:] # lower right block
                C = cov[:a_len, a_len:] # upper right block
                b = wind_meas[:,j]
                mu, sigma = cond_gaussian(b, mu_a, mu_b, A, B, C)
                
            else:
                mu = np.zeros(len(locs))
                sigma = cov
    
                # For numerical reasons, clip the minimum value to 0
                # sigma = np.clip(sigma, 0, np.inf)

            # For numerical reasons, round to nearest PSD matrix
            sigma = roundPSD(sigma)

            # These are draws for the hidden locations num_draws x a_len              
            comp_draws = multivariate_normal(mu, sigma, allow_singular=True).rvs(size=num_draws)
                  
            # To handle edge cases like num_draws = 1, or only have 1 hidden location
            while len(comp_draws.shape) < 2:
                comp_draws = np.expand_dims(comp_draws, axis=0)
            
            temp = W_draws[i, :, :, :, j] # num_draws x grid_size[0] x grid_size[1]
            
            # TODO: Do these operations in a single-step
            for k in range(len(meas_locs)):
                temp[:, meas_locs[k,0], meas_locs[k,1]] = wind_meas[k, j]
                
            for k in range(len(hidden_locs)):
                try:
                    temp[:, hidden_locs[k,0], hidden_locs[k,1]] = comp_draws[:, k]
                except:
                    print('comp_draws', comp_draws)
                    print('hidden_locs[k]', hidden_locs[k])
                    
    return gamma_draws, W_draws

if __name__ == '__main__':
    # n, m (# x vals, # y vals)
    grid_size = (8, 8)
    gamma = 0.1
    W, dists, locs = create_wind_vector(grid_size, gamma) 
    ax, _ = plot_wind_vector_and_speeds(W, alpha=0.5, color='black')
    ax.set_title('True Wind Field')
    
    num_disc = 50
    gamma_vals = np.linspace(0.01, 1, num_disc, endpoint=True)
    # Uniform initialization of gamma distribution i.e. uniform prior on gamma
    # Use log pdf
    gamma_dist = np.log(1/num_disc * np.ones(num_disc))
    
    all_wind_meas = np.reshape(W, (-1,2))
    
    # How many gamma draws to take from the gamma posterior
    num_gamma = 10
    # How many wind fields to generate for each gamma draw
    num_draws = 10
   
    new_dist_rec = np.copy(gamma_dist)
    old_locs = None
    old_wind_meas = None
    for i, wind_meas in enumerate(all_wind_meas):
        # Try single-shot posterior updating
        new_locs = locs[:i+1]
        new_wind_meas = all_wind_meas[:i+1,:]
        new_dist = update_gamma_dist(gamma_vals, gamma_dist, new_wind_meas, new_locs) # QUESTION: NOT NEEDED?
    
        # Try recursive posterior updating
        new_dist_rec = update_gamma_dist(gamma_vals, new_dist_rec, 
            np.expand_dims(wind_meas, axis=0), np.expand_dims(locs[i], axis=0), 
            old_wind_meas, old_locs)
          
        gamma_draws, W_draws = sample_posterior_wind_vector(grid_size, gamma_vals, new_dist_rec, 
                                         num_gamma, num_draws, new_wind_meas, new_locs)
        
        if i % 10 == 0:
            plt.figure()
            plt.plot(gamma_vals, belief_to_dist(new_dist), label='Single-Shot')
            plt.plot(gamma_vals, belief_to_dist(new_dist_rec), label='Recursive')
            plt.title(r'Posterior over $\gamma$ with ' + str(i+1) + ' Measurements')
            plt.xlabel(r'$\gamma$')
            plt.ylabel(r'Posterior Density')
            plt.legend()
        
            # Visualize what the average wind field completion looks like
            W_list = W_draws.reshape((-1, grid_size[0], grid_size[1], 2))
            W_avg = np.mean(W_list, axis=0)
            ax, _ = plot_wind_vector_and_speeds(W_avg, alpha=0.5, color='black')
            ax.set_title('Average Wind Field with ' + str(i+1) + ' Measurements')
            
        if i > 0:
            old_locs = np.vstack([old_locs, new_locs[i]])
            old_wind_meas = np.vstack([old_wind_meas, wind_meas])
        else:
            old_locs = np.expand_dims(new_locs[i], axis=0)
            old_wind_meas = np.expand_dims(wind_meas, axis=0)
     


