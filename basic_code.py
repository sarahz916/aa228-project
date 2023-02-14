# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:24:01 2023

@author: zousa
"""
import numpy as np

class Environment:
    def __init__(self, env):
        self.env = env
    
    def perceived_wind(self, raw_wind, curr_loc):
        # Find out if there are buildings adjacent to current location
        curr_x, curr_y = curr_loc
        p_w = raw_wind
        m, n = self.env.shape
    
        if curr_x - 1 >= 0 and self.env[curr_x - 1, curr_y] == 1:
            if raw_wind[0] > 0: # wind blowing left to right
                p_w[0] = 0
        if curr_x + 1 < n and self.env[curr_x + 1, curr_y] == 1:
            if raw_wind[0] < 0: # wind blowing to the left 
                p_w[0] = 0
        if curr_y - 1 >= 0 and self.env[curr_x, curr_y - 1] == 1:
            if raw_wind[1] > 0: # wind blowing up
                p_w[1] = 0
        if curr_y + 1 < m and self.env[curr_x, curr_y + 1] == 1:
            if raw_wind[1] < 0: # wind blowing down
                p_w[1] = 0
        
        # check the four corner buildings
        # bottom left
        if np.all(curr_loc != [0,0]) and self.env[curr_x - 1, curr_y -1] == 1:
            if np.all(raw_wind > [0, 0]):
                p_w = np.array([0, 0])
        # top left
        if np.all(curr_loc != [0,m - 1]) and self.env[curr_x - 1, curr_y + 1] == 1:
            if raw_wind[0] > 0 and raw_wind[1] < 0:
                p_w = np.array([0, 0])
        # top right
        if np.all(curr_loc != [n - 1 ,m - 1]) and self.env[curr_x + 1, curr_y + 1] == 1:
            if np.all(raw_wind < [0, 0]):
                p_w = np.array([0, 0])
        # bottom right
        if np.all(curr_loc != [n - 1,0]) and self.env[curr_x + 1, curr_y - 1] == 1:
            if raw_wind[0] < 0 and raw_wind[1] > 0:
                p_w = np.array([0, 0])
        return p_w
    
    def one_step_cost(self, curr_loc, move, all_wind):
        p_w = self.perceived_wind(all_wind[tuple(curr_loc)], curr_loc)
        cost = -np.inner(p_w, move)
        return cost
    
    def path_cost(self, start_loc, moves, all_wind):
        total_cost = 0
        curr_loc = np.copy(start_loc)
        for m in moves:
            total_cost += self.one_step_cost(curr_loc, m, all_wind)
            curr_loc = curr_loc + m
        return total_cost