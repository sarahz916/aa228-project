import numpy as np
import matplotlib.pyplot as plt

### Plotting Helpers ###

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

######

class Environment:
    def __init__(self, occ, base_cost, min_cost, buildings_block=True):
        self.occ = occ
        # What is the baseline cost of a given step
        self.base_cost = base_cost
        # Put a cap on the minimum cost
        self.min_cost = min_cost
        # Set to True if want buildings to block the wind else False
        self.buildings_block = buildings_block
        
    def perceived_wind(self, all_wind, curr_loc):
        # Find out if there are buildings adjacent to current location
        curr_x, curr_y = curr_loc
        raw_wind = all_wind[curr_x, curr_y]
        # important to copy so don't change the wind field itself
        p_w = np.copy(raw_wind)
        if self.buildings_block == False:
            return p_w
        n, m = self.occ.shape
    
        if curr_x - 1 >= 0 and self.occ[curr_x - 1, curr_y] == 1:
            if raw_wind[0] > 0: # wind blowing left to right
                p_w[0] = 0
        
        if curr_x + 1 < n and self.occ[curr_x + 1, curr_y] == 1:
            if raw_wind[0] < 0: # wind blowing to the left 
                p_w[0] = 0
        if curr_y - 1 >= 0 and self.occ[curr_x, curr_y - 1] == 1:
            if raw_wind[1] > 0: # wind blowing up
                p_w[1] = 0
        if curr_y + 1 < m and self.occ[curr_x, curr_y + 1] == 1:
            if raw_wind[1] < 0: # wind blowing down
                p_w[1] = 0
                
        return p_w
    
    def perceived_wind_x(self, all_wind, curr_loc):
        curr_x, curr_y = curr_loc
        raw_wind = all_wind[curr_x, curr_y]
        p_w_x = raw_wind[0]
        block_left, block_right = False, False
        
        n, m = self.occ.shape
        if curr_x - 1 >= 0 and self.occ[curr_x - 1, curr_y] == 1:
            if raw_wind[0] > 0: # wind blowing left to right
                block_left = True
        
        if curr_x + 1 < n and self.occ[curr_x + 1, curr_y] == 1:
            if raw_wind[0] < 0: # wind blowing to the left 
                block_right = True
        
        if block_left and block_right:
            return None, 0
        elif block_left or block_right:
            return None, np.sign(p_w_x)
        else:
            return p_w_x, np.sign(p_w_x)
                
    def perceived_wind_y(self, all_wind, curr_loc):
        curr_x, curr_y = curr_loc
        raw_wind = all_wind[curr_x, curr_y]
        p_w_y = raw_wind[1]
        block_up, block_down = False, False
        
        n, m = self.occ.shape
        if curr_y - 1 >= 0 and self.occ[curr_x, curr_y - 1] == 1:
            if raw_wind[1] > 0: # wind blowing up
                block_down = True
        if curr_y + 1 < m and self.occ[curr_x, curr_y + 1] == 1:
            if raw_wind[1] < 0: # wind blowing down
                block_up = True
        
        if block_down and block_up:
            return None, 0
        elif block_down or block_up:
            return None, np.sign(p_w_y)
        else:
            return p_w_y, np.sign(p_w_y)
        
    def one_step_cost(self, curr_loc, move, all_wind):
        p_w = self.perceived_wind(all_wind, curr_loc)
        wind_cost = -np.inner(p_w, move)
        cost = max(self.min_cost, self.base_cost + wind_cost)
        return cost
    
    def path_cost(self, start_loc, moves, all_wind):
        total_cost = 0
        edge_costs = []
        curr_loc = np.copy(start_loc)
        for move in moves:
            step_cost = self.one_step_cost(curr_loc, move, all_wind)
            total_cost += step_cost
            edge_costs.append(step_cost)
            curr_loc += move
        return total_cost, edge_costs
        
    def get_avg_cost(self, curr_loc, move, all_wind_list):
        costs = [self.one_step_cost(curr_loc, move, all_wind) for
                 all_wind in all_wind_list]
        return np.mean(costs), costs
    
    