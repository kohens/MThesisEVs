import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import pandas as pd

#torch.autograd.set_detect_anomaly(True)

# Inputs to provide:
    # array with forecasted electricity price and forecasted pv generation
    # array with arrival and departure times of EVs (real values)
    # array with required energy and max power of each loading session
    # array with loading stations of each loading session
    



# Help functions

limits = np.load('state_limits_N_ev_16_Lc_30.npz')
limits = limits['limits']

def av(Array, time1, time2):
    if time1 >= time2:
        return Array[time2]
    else: 
        return np.mean(Array[time1:time2])

def scaling(data, max_array):
    return 2*data/max_array - 1

def av_vec(Array, time1, time2):
    output = np.zeros(len(time1))
    for j in range(len(time1)):
        t1 = time1[j]
        t2 = time2[j]
        if t1 >= t2:
            output[j] = Array[t2]
        else: output[j] = np.mean(Array[t1:t2])
    return output


# Get the aggregated fleet state Z from vectors with required energy of the fleet EVs and the power limit of the EVs
def get_fleet_state(E, P):
    return np.array((sum(E), sum(P)))

# Function to obtain real-time data from the environment
def create_time_function(time, data):
    return  interp1d(time, data, kind='linear')

# Return a vector with vehicles and their original required energy level
def get_EV_fleet(t, IDs, E_req, P_lim, T_arr, T_dep, Stations, StationVector, L_c): 
    vec = np.where(t * np.ones(len(T_arr)) + L_c / 60 >= T_arr, np.ones(len(T_arr)), np.zeros(len(T_arr)))
    vec = np.where(t * np.ones(len(T_arr)) <= T_dep, vec, np.zeros(len(T_arr)))

    IDs = IDs[vec != 0]
    T_arr = T_arr[vec != 0]
    T_dep = T_dep[vec != 0]
    P_lim = P_lim[vec != 0]
    E_req = E_req[vec != 0]
    Stations = Stations[vec != 0]

    IDvec = np.sum((StationVector[:, None] == Stations) * IDs, axis=1)
    T_arrvec = np.sum((StationVector[:, None] == Stations) * T_arr, axis=1)
    T_depvec = np.sum((StationVector[:, None] == Stations) * T_dep, axis=1)
    P_limvec = np.sum((StationVector[:, None] == Stations) * P_lim, axis=1)
    E_reqvec = np.sum((StationVector[:, None] == Stations) * E_req, axis=1)
    
    return IDvec, E_reqvec, P_limvec, T_arrvec, T_depvec
    # Return a vector with the IDs of all connected EVs and their original required energy level, their power limit and arrival and departure times
    # For each charging station, if no EV present at charging station -> all values 0
    
    

# Heuristic dispatch of aggregated power of different loading stations
def heuristic_dispatch(P, P_min, P_max, delta_T_dep, E_req, P_lim, N_ev):
    flexibility = delta_T_dep - np.ones(N_ev) - np.divide(E_req * 60, P_lim)
    sorted_indices = np.argsort(flexibility)
    P_b = np.zeros(N_ev)

    # First ensure that all vehicles get their minimum required energy
    indices = sorted_indices[P_max[sorted_indices] > P_min[sorted_indices]]  # Filter only the relevant indices
    min_energy_needed = np.minimum(P, P_min[indices])
    P_b[indices] = min_energy_needed
    P -= np.sum(min_energy_needed)

    # Further heuristic dispatch of remaining power
    remaining_indices = sorted_indices[P_max[sorted_indices] <= P_min[sorted_indices]]
    remaining_power = P
    while remaining_power > 0 and np.sum(P_b) < np.sum(E_req) and remaining_indices != []:
        remaining_indices = sorted_indices[P_max[sorted_indices] > P_b[sorted_indices]]  # Filter only the remaining indices
        extra_power = np.minimum(P_max[remaining_indices] - P_b[remaining_indices], remaining_power)
        P_b[remaining_indices] += extra_power
        remaining_power -= np.sum(extra_power)
    return P_b

# Backup controller to make sure the power supplied suffices te requirements
def backup_controller(P, P_min, P_max):
    P_min = P_min*np.ones(len(P))
    P_max = P_max*np.ones(len(P))
    P = np.where(P >= P_max, P_max, P)
    P = np.where(P <= P_min, P_min, P)
    return P


def real_time_controller(P, M_start, M_end, L_c, N_ev, P_lim, t, k, pv_generation):
    EVs = np.where(P > 0)[0]
    P_r = np.zeros((N_ev, L_c))
    
    P_r_max = np.minimum(P_lim, P * k)
    P_r_min = P / k
    E_rem = P * (M_end - M_start) / 60
    
    for m in range(L_c):
        P_sum_max_m = 0
        P_sum_min_m = 0
        P_min_m = np.zeros(N_ev)
        P_max_m = np.zeros(N_ev)
        EVsThisMinute = [i for i in EVs if M_start[i] <= m < M_end[i]]
        for j in EVsThisMinute:
            P_max_m[j] = min(E_rem[j]*60, P_r_max[j])
            P_sum_max_m += P_max_m[j]
            P_min_m[j] = np.clip(E_rem[j]*60-(M_end[j]-m-1)*P_r_max[j], P_r_min[j], P_max_m[j])
            P_sum_min_m += P_min_m[j]
        for j in EVsThisMinute:
            if P_sum_max_m == P_sum_min_m:
                P_r[j][m] = P_min_m[j]
            else:
                P_pv = pv_generation(t+m/60)
                P_r[j][m] = np.clip(P_min_m[j]+(P_pv-P_sum_min_m)*(P_max_m[j]-P_min_m[j])/(P_sum_max_m-P_sum_min_m), P_min_m[j], P_max_m[j])
            E_rem[j] -= P_r[j][m]/60

    return P_r

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    centered_data = data - mean
    normalized_data = centered_data / (std + 1e-8)
    return normalized_data

# Step 1: Environment class

class Environment(gym.Env):
    
    def __init__(self, start_time, end_time, N_ev, forecast, price_forecast, N_past, N_fut, L_c, E_reqs, P_max, T_arr, T_dep, Stations, StationVector, IDs, kappa, injection_price,consumption_price, pv_generation ):
        # Define your environment-specific initialization here
        # Set up the necessary variables, observation space, and action space
        self.pv_forecast = forecast
        self.price_forecast = price_forecast
        self.N_past = N_past
        self.N_fut = N_fut
        self.L_c = L_c
        self.start_time = start_time
        self.end_time = end_time
        self.t = 0
        self.N_ev = N_ev
        self.E_reqs = E_reqs
        self.P_max = P_max
        self.T_arr = T_arr
        self.T_dep = T_dep
        self.Stations = Stations
        self.StationVector = StationVector
        self.IDs = IDs
        self.kappa = kappa
        self.E_req = None
        #self.EV_IDs = None
        self.EV_levels = None
        self.P_lim = None
        self.P_pv_past = None
        # Functions return the injection/consumption price at a time t
        self.injection_price = injection_price
        self.consumption_price = consumption_price
        
        #Function returns the pv generation at a time t
        self.pv_generation = pv_generation

        
    # Set the environment in the initial state
    def reset(self):
        # Reset the environment to its initial state
        # Return the initial observation
        self.t = self.start_time
        self.E_req = np.zeros(self.N_ev)
        #self.EV_IDs = np.zeros(self.N_ev)
        self.P_lim = np.zeros(self.N_ev)
        self.EV_levels = np.zeros(self.N_ev)
        self.T_arr_fleet = np.zeros(self.N_ev)
        self.T_dep_fleet = np.zeros(self.N_ev)
        self.IDs_fleet =  np.zeros(self.N_ev)
        
        [self.E_req, self.P_lim, self.IDs_fleet, self.EV_levels, self.T_arr_fleet, self.T_dep_fleet] = self.update_fleet()
        self.P_pv_past = np.zeros(int(self.N_past*60//self.L_c))
        state = self.initialize_state()
        return state
    
    # Update the environment given a certain action
    def step(self, action):
        # Take a step in the environment based on the given action
        # Update the state, compute the reward, and check if the episode is done
        # Return the next observation, reward, done flag, and additional information
        #print(self.E_req)
        # Update of the required energy of the EV-fleet
        [P_charge, P_min, P_max] = self.divide_aggregated_power(action)
        P_charge = backup_controller(P_charge, P_min, P_max)
        #print(P_charge)
        [M_start, M_end] = self.get_M_vectors()
        E_charged = np.multiply((M_end-M_start)/60, P_charge)
        self.E_req -= E_charged
        self.E_req = np.maximum(self.E_req, np.zeros(self.N_ev))
        P_charge = real_time_controller(P_charge, M_start, M_end, self.L_c, self.N_ev, self.P_lim, self.t, self.kappa, self.pv_generation)
        #print(P_charge)
        [self.E_req, self.P_lim, self.IDs_fleet, self.EV_levels, self.T_arr_fleet, self.T_dep_fleet] = self.update_fleet()
        # Calculation of the reward
        reward = self.calculate_reward(P_charge)
        
        # Check if episode is done: if time = end_time
        done = False
        self.t += self.L_c/60
        if self.t > self.end_time-self.L_c/60:
            done = True
            self.done_procedure()
        
        # Calculation of new state
        # Updates that do not depend on the action
        P_pv_past = self.get_past_PV()
        P_pv_current = self.get_and_update_PV()
        P_pv_state_t = np.append(P_pv_past,P_pv_current)
        P_f_pv_state_t = self.get_forecast_tuple()
        L_cons = self.get_price_tuple()
        Z_t = get_fleet_state(self.E_req, self.P_lim)
        state = np.concatenate((Z_t, (self.t % 24)*np.ones(1), P_pv_state_t, P_f_pv_state_t, L_cons))

        return state, reward, done
    
    # Initialize the state vector
    def initialize_state(self):
        # Z_t = aggregated fleet state: total required energy and total power limit
        Z_t  = get_fleet_state(self.E_req, self.P_lim)
        
        # P_pv_state_t, P_f_pv_state_t: current and forecasted value of the PV generation at time t
        P_pv_past = self.get_past_PV()
        P_pv_current = self.get_and_update_PV()

        P_pv_state_t = np.append(P_pv_past,P_pv_current)

        
        P_f_pv_state_t = self.get_forecast_tuple()
        
        #Price component of the state
        L_cons = self.get_price_tuple()
        return np.concatenate((Z_t, (self.t % 24)*np.ones(1), P_pv_state_t, P_f_pv_state_t, L_cons))

    
    #Update the currently connected EVs and their properties (E_req, P_lim, T_arr, T_dep)
    def update_fleet(self):
        [IDs, levels, max_power, T_arr, T_dep] = get_EV_fleet(self.t, self.IDs,self.E_reqs, self.P_max, self.T_arr, self.T_dep, self.Stations, self.StationVector, self.L_c)
        return np.where(self.IDs_fleet == IDs, self.E_req, levels), max_power, IDs, levels, T_arr, T_dep
    
    # Get the past PV component of the state at time with hourly averages for the past N_past hours
    def get_past_PV(self):
        indices1 = -1*np.arange(self.N_past) * 60 // self.L_c - np.ones(self.N_past, dtype=int)
        indices2 = -1*(np.arange(self.N_past) + np.ones(self.N_past, dtype=int)) * 60 // self.L_c - np.ones(self.N_past, dtype=int)
        P_pv_past = av_vec(self.P_pv_past, indices2, indices1)
        return P_pv_past
    
    # Get current PV generation and add to historic generation array
    def get_and_update_PV(self):
        current_PV = self.pv_generation(self.t)
        self.P_pv_past = np.concatenate((self.P_pv_past[1:],np.array([current_PV])))
        return current_PV
    
    # Get state tuple of the PV generation forecast
    def get_forecast_tuple(self):
        indices_past1 = ((self.t - self.start_time) * 60 / self.L_c - np.arange(self.N_past + 1) * 60 / self.L_c).astype(int)
        indices_past2 = indices_past1[0:-1]
        indices_past1 = indices_past1[1:]
        indices_past1 = np.maximum(indices_past1, 0)
        indices_past2 = np.maximum(indices_past2, 0)
        end = len(self.pv_forecast) - 1
        indices_past1 = np.minimum(indices_past1, end)
        indices_past2 = np.minimum(indices_past2, end)
        P_f_past = av_vec(self.pv_forecast, indices_past1, indices_past2)
    
        indices_fut1 = ((self.t - self.start_time) * 60 / self.L_c + np.arange(self.N_fut) * 60 / self.L_c).astype(int)
        indices_fut2 = ((self.t - self.start_time) * 60 / self.L_c + np.arange(1, self.N_fut + 1) * 60 / self.L_c).astype(int)
        indices_fut1 = np.maximum(indices_fut1, 0)
        indices_fut2 = np.maximum(indices_fut2, 0)
        indices_fut1 = np.minimum(indices_fut1, end)
        indices_fut2 = np.minimum(indices_fut2, end)
        P_f_fut = av_vec(self.pv_forecast, indices_fut1, indices_fut2)
    
        index_rest = int((self.t - self.start_time)*60/self.L_c + self.N_fut*60/self.L_c + 1)
        P_f_rest = av(self.pv_forecast, index_rest, len(self.pv_forecast) - 1)
        return np.concatenate((P_f_past, P_f_fut, np.array([P_f_rest])))

    
    # Get state tuple of the price forecast
    def get_price_tuple(self):
        L_t = np.array([self.consumption_price(self.t)])
    
        indices_fut1 = ((self.t - self.start_time) * 60 / self.L_c + np.arange(self.N_fut) * 60 / self.L_c).astype(int)
        indices_fut2 = ((self.t - self.start_time) * 60 / self.L_c + np.arange(1, self.N_fut + 1) * 60 / self.L_c).astype(int)
        indices_fut1 = np.maximum(indices_fut1, 0)
        indices_fut2 = np.maximum(indices_fut2, 0)
        end = len(self.price_forecast) - 1
        indices_fut1 = np.minimum(indices_fut1, end)
        indices_fut2 = np.minimum(indices_fut2, end)
        L_f_fut = av_vec(self.price_forecast, indices_fut1, indices_fut2)
    
        index_rest = int((self.t + self.N_fut - self.start_time) * 60 / self.L_c) + 1
        L_f_rest = av(self.price_forecast, index_rest, len(self.price_forecast) - 1)

        return np.concatenate((L_t, L_f_fut, np.array([L_f_rest])))
    
    # Update of the required energies of each charge station
    def divide_aggregated_power(self, action):
        delta_T_dep = self.T_dep_fleet-self.t*np.ones(self.N_ev)
        # If value in delta_T_dep < 0: no EV present at that location
        [M_start, M_end] = self.get_M_vectors()
        P_b_t_min = np.divide((self.E_req - np.multiply((delta_T_dep-1), self.P_lim))*60, M_end-M_start)
        P_b_t_min = np.where((M_end - M_start) != 0, P_b_t_min, 0)
        P_b_t_min = np.maximum(np.zeros(len(P_b_t_min)), P_b_t_min)

        P_b_t_max = np.divide(60 * self.E_req, M_end - M_start)
        P_b_t_max = np.where((M_end - M_start) != 0, P_b_t_max, 0)
        P_b_t_max = np.minimum(self.P_lim, P_b_t_max)
        P_b_t = heuristic_dispatch(action, P_b_t_min, P_b_t_max, delta_T_dep, self.E_req, self.P_lim, self.N_ev)
        return P_b_t, P_b_t_min, P_b_t_max
    
    # Get vectors with first and last minute in which each EV is present in time frame t:t+1
    def get_M_vectors(self):

        EV_present = np.where(self.T_arr_fleet != np.zeros(self.N_ev), np.ones(self.N_ev), np.zeros(self.N_ev))
        End_minutes = np.multiply(EV_present, ((self.T_dep_fleet-self.t*np.ones(self.N_ev))*60).astype(int)+np.ones(self.N_ev, dtype=int))
        M_end = np.where(self.T_dep_fleet >= self.t+self.L_c/60, self.L_c*np.ones(self.N_ev), End_minutes)
        Start_minutes = np.multiply(EV_present, ((self.t*np.ones(self.N_ev))*60-self.T_arr_fleet).astype(int)+np.ones(self.N_ev, dtype=int))
        M_start = np.where(self.T_arr_fleet >= self.t*np.ones(self.N_ev), Start_minutes, np.zeros(self.N_ev))
        return M_start, M_end
    
    # Calculate the cost of a charging plan
    def calculate_reward(self, P_charge):
        P_diff = np.sum(P_charge, axis=0) - self.pv_generation(self.t + np.arange(self.L_c) / 60)
        consumption_prices = self.consumption_price(self.t + np.arange(self.L_c) / 60)
        injection_prices = self.injection_price(self.t + np.arange(self.L_c) / 60)
    
        cost = np.sum(np.where(P_diff >= 0, P_diff * consumption_prices, P_diff * injection_prices)) / 60
        return -cost
    
    def done_procedure(self):
        keep = self.T_dep > self.end_time
        self.T_arr = self.T_arr[keep]
        self.T_dep = self.T_dep[keep]
        self.E_reqs = self.E_reqs[keep]
        self.P_max = self.P_max[keep]
        self.IDs = self.IDs[keep]
        self.Stations = self.Stations[keep]
        # Go to a new epoch, move the data of the next week to the front
        self.T_arr -= (self.end_time-1)*np.ones(len(self.T_arr))
        self.T_dep -= (self.end_time-1)*np.ones(len(self.T_dep))
        
# Step 2: Initialize the policy network

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_size, output_size):
       super(ActorCriticPolicy, self).__init__()
       self.shared_layer = nn.Linear(input_size, 64)
       
       self.actor_layer1 = nn.Linear(64, 32)
       self.actor_layer2 = nn.Linear(32, 32)
       self.actor_output = nn.Linear(32, output_size)
       
       self.critic_layer1 = nn.Linear(64, 32)
       self.critic_layer2 = nn.Linear(32, 32)
       self.critic_output = nn.Linear(32, 1)
       self.critic_scaling = nn.Linear(1, 1)

    def forward(self, x):
       x =  torch.tanh(self.shared_layer(x))
       
       actor_x =  torch.tanh(self.actor_layer1(x))
       actor_x =  torch.tanh(self.actor_layer2(actor_x))
       actor =  torch.tanh(self.actor_output(actor_x))
       
       critic_x =  torch.tanh(self.critic_layer1(x))
       critic_x =  torch.tanh(self.critic_layer2(critic_x))
       critic =  self.critic_output(critic_x)
       #critic = self.critic_scaling(critic_x)

       return actor, critic
   
    def denormalize_action(self, normalized_action, a_max):
        denormalized_action = (normalized_action + 1) * 0.5 * (a_max)
        return denormalized_action
    
    
    
    def denormalize_value(self, normalized_value):
        # Inverse tanh to denormalize the value
        denormalized_value = normalized_value
        return denormalized_value


# Step 3: Proximal policy optimization agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, env, epochs, batch_size, epsilon, value_coeff, entropy_coeff, clip_epsilon, lr, discount_factor, lamda):
        self.env = env
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.discount_factor = discount_factor
        self.lamda = lamda
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = ActorCriticPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        
    def collect_data(self):
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        P_batch = []
        normalized_rewards_batch = []
    
        for i in range(self.batch_size):
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            P = []
            state = self.env.reset()
            done = False
            while not done:
                P_lim_agg = state[1]
                P_lim_agg = limits[1]
                state = scaling(state, limits)
                action = self.select_action(state, P_lim_agg)
                next_state, reward, done = self.env.step(action)
                P.append(P_lim_agg)
                states.append(state)
                actions.append(action)
                rewards.append(reward)  # Append the actual reward value
                next_states.append(scaling(next_state, limits))
                dones.append(done)
                state = next_state  
            #print(P)
            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)
            P_batch.append(P)
            normalized_rewards_batch.append(normalize_data(rewards))
        print(np.sum(rewards_batch)/self.batch_size)
        #print(actions_batch)
        # Convert the batched data to PyTorch tensors
        states_batch = torch.tensor(states_batch, dtype=torch.float32)
        actions_batch = torch.tensor(actions_batch, dtype=torch.float32)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
        next_states_batch = torch.tensor(next_states_batch, dtype=torch.float32)
        dones_batch = torch.tensor(dones_batch, dtype=torch.float32)
        P_batch = torch.tensor(P_batch, dtype=torch.float32)
        #print(P_batch)
        normalized_rewards_batch = torch.tensor(normalized_rewards_batch, dtype=torch.float32)
        return states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, P_batch, normalized_rewards_batch

    
    def select_action(self, state, Limit):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            actor_output = self.network(state)[0]
        # Exploration noise
        std_dev = 0.01
        noise = torch.randn_like(actor_output)*std_dev
        action_tensor = actor_output + noise
        normalized_action = action_tensor.item()
        action = self.network.denormalize_action(normalized_action, Limit)
        return action

    def calculate_advantages_and_targets(self, rewards_batch, next_states_batch, dones_batch, raw_rewards_batch):
        
        values_batch = self.network(next_states_batch)[1].detach().squeeze()

        advantages_batch = []
        target_values_batch = []
        
        for traj_idx in range(self.batch_size):
            values = values_batch[traj_idx]
            rewards = rewards_batch[traj_idx]
            dones = dones_batch[traj_idx]
            raw_rewards = raw_rewards_batch[traj_idx]
        
            mu_values = values.mean()
            sigma_values = values.std()
            normalized_values = (values - mu_values) / sigma_values
        
            # Compute the target values using the Bellman equation without iteration
            target_values = rewards + self.discount_factor * (1 - dones) * torch.roll(normalized_values, -1)
            target_values[-1] = 0  # Set the last target value to 0 since there's no next state
            
            # Compute the temporal difference
            delta = rewards + self.discount_factor * (1 - dones) * torch.roll(normalized_values, -1) - normalized_values
            #print(delta.shape)
            T = len(rewards)
            # Calculate the GAE with tensor operations
            gae = torch.zeros_like(rewards)
            gae[-1] = delta[-1]
            for i in reversed(range(T - 1)):
                gae[i] = delta[i] + self.discount_factor * self.lamda * gae[i + 1]
            advantages = gae.float()
            advantages[T - 1] = rewards[T - 1] - normalized_values[T - 1]
        
            advantages_batch.append(advantages)
            target_values_batch.append(target_values)
        
        advantages_batch = torch.stack(advantages_batch)
        target_values_batch = torch.stack(target_values_batch)
        
        return advantages_batch, target_values_batch

    def update_policy(self, old_log_probs, epoch, old_network):
        self.optimizer.zero_grad()
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, P_batch, normalized_rewards_batch = self.collect_data()
        # Calculate advantages and target values using the modified function
        advantages_batch, target_values_batch = self.calculate_advantages_and_targets(normalized_rewards_batch, next_states_batch, dones_batch, rewards_batch)
        P_batch = P_batch#[indices]

        old_actor_outputs = old_network(states_batch)[0].detach()
        std_dev_batch = 0.5 * P_batch+0.000001
        denormalized_old_outputs = (old_actor_outputs.squeeze() + 1) * 0.5 * P_batch.squeeze()

        old_log_probs_batch = self.calculate_log_probs(denormalized_old_outputs, actions_batch, std_dev_batch).detach()
    
        old_network = ActorCriticPolicy(self.state_dim, self.action_dim)
        old_network.load_state_dict(self.network.state_dict())
        actor_outputs_batch, critic_outputs_batch = self.network(states_batch)
        denormalized_actor_outputs = (actor_outputs_batch.squeeze(2) + 1) * 0.5 * P_batch
        mu_values = critic_outputs_batch.mean()
        sigma_values = critic_outputs_batch.std()
        normalized_critic_outputs = (critic_outputs_batch - mu_values) / sigma_values
    
        value_loss_batch = F.mse_loss(normalized_critic_outputs.squeeze(), target_values_batch.squeeze())

        new_log_probs_batch = self.calculate_log_probs(denormalized_actor_outputs, actions_batch, std_dev_batch)
        log_ratios_batch = new_log_probs_batch - old_log_probs_batch
        ratios_batch = torch.exp(log_ratios_batch)
    
        clipped_ratios_batch = torch.clamp(ratios_batch, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        surr1_batch = ratios_batch * advantages_batch
        surr2_batch = clipped_ratios_batch * advantages_batch
        policy_loss_batch = -torch.min(surr1_batch, surr2_batch).mean()
        total_loss_batch = policy_loss_batch + self.value_coeff * value_loss_batch
        total_loss_batch.backward()
        self.optimizer.step()
    
        return policy_loss_batch, value_loss_batch, rewards_batch, actions_batch, states_batch, new_log_probs_batch.detach(), old_network
    
    
    def train(self):
        policy_losses = []
        value_losses = []
        rewards_list = []
        actions_list = []
        states_list = []
        old_log_probs = 0
        old_network = ActorCriticPolicy(self.state_dim, self.action_dim)
        old_network.load_state_dict(self.network.state_dict())
    
        for epoch in range(self.epochs):
            print(epoch)
            policy_loss, value_loss, rewards_batch, actions_batch, states_batch, old_log_probs, old_network = self.update_policy(old_log_probs, epoch, old_network)
            
            policy_loss = policy_loss.detach()
            value_loss = value_loss.detach()
            rewards = rewards_batch.detach()
            actions = actions_batch.detach()
            states = states_batch.detach()
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            rewards_list.append(rewards_batch)
            actions_list.append(actions_batch)
            states_list.append(states_batch)
    
        max_length = max(len(rewards) for rewards in rewards_list)
        padded_rewards = torch.stack(
            [torch.nn.functional.pad(r, (0, max_length - len(r))) for r in rewards_list]
        )
        padded_actions = torch.stack(
            [torch.nn.functional.pad(a, (0, max_length - len(a))) for a in actions_list]
        )
        padded_states = torch.stack(
            [torch.nn.functional.pad(a, (0, max_length - len(a))) for a in states_list]
        )
    
        PL = policy_losses
        VL = value_losses
        R = padded_rewards.numpy()
        A = padded_actions.numpy()
        S = padded_states.numpy()
    
        return PL, VL, R, A, S, self.network

            
    def calculate_cost(self, env):
        # Collect total charging plan as well to visualize
        cost = 0
        costs = []
        actions = []
        state = self.env.reset()
        done = False
        while not done:
            action, _ = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            state = next_state
            cost -= reward
            costs.append(reward)
            actions.append(action)
        return cost, -1*costs
    
    def calculate_entropy(self, states):
        logits, _ = self.network(states)
        prob_dist = F.softmax(logits, dim=-1)
        entropy = -(prob_dist * torch.log(prob_dist)).sum(dim=-1).mean()
        return entropy
    
    def calculate_log_probs(self, mean, actions, std_dev):
        #print(mean)
        #print(actions)
        #print(std_dev)
        action_distribution = torch.distributions.Normal(mean, std_dev)
        #print(std_dev)
        # Calculate the log probability of the action given the Gaussian distribution
        log_probs = action_distribution.log_prob(actions)
        #print(log_probs)
        
        return log_probs

# Business as usual charging
class BAUCharging:
    def __init__(self, start_time, end_time, E_reqs, P_max, T_arr, T_dep, time_step, injection_price,consumption_price, pv_generation):
        # Define your environment-specific initialization here
        # Set up the necessary variables, observation space, and action space
        self.time_step = time_step
        self.end_time = end_time
        self.t = start_time
        self.P_max = P_max
        self.charge_start = T_arr
        
        # Functions return the injection/consumption price at a time t
        self.injection_price = injection_price
        self.consumption_price = consumption_price
        
        #Function returns the pv generation at a time t
        self.pv_generation = pv_generation
        
        # Determine the time values at which charging stops (battery full or vehicle departs)
        self.charge_stop = get_charge_end(E_reqs,P_max, T_arr, T_dep)
        
    def calculate_cost(self):
        cost = 0
        i = 0
        PV = np.zeros(int((self.end_time-self.t)//self.time_step)+1)
        costs = np.zeros(int((self.end_time-self.t)//self.time_step)+1)
        price = np.zeros(int((self.end_time-self.t)//self.time_step)+1)
        P_cons = np.zeros(int((self.end_time-self.t)//self.time_step)+1)
        while self.t <= self.end_time:
            P = get_power_consumption(self.t, self.P_max, self.charge_start, self.charge_stop)
            P_pv = self.pv_generation(self.t)
            P_diff = P - P_pv
            if P_diff <= 0:
                c = P_diff*self.injection_price(self.t)*self.time_step
            else:
                c = P_diff*self.consumption_price(self.t)*self.time_step
            cost += c
            P_cons[i] = P
            costs[i] = cost
            price[i] = c
            PV[i] = P_pv
            self.t += self.time_step
            i += 1
        return cost, costs, price, P_cons, PV

        
def get_charge_end(E, P, T_arr, T_dep):
    T_needed = np.divide(E, P)
    T_fully_charged = T_arr + T_needed
    T_end = np.where(T_dep <= T_fully_charged, T_dep, T_fully_charged)
    return T_end

def get_power_consumption(t, P, Start, Stop):
    ActiveSessions = np.where(t*np.ones(len(Start)) >= Start, P, np.zeros(len(Start)))
    ActiveSessions = np.where(t*np.ones(len(Start)) <= Stop, ActiveSessions, np.zeros(len(Start)))
    return np.sum(ActiveSessions)

class PIOSolution:
    def __init__(self, start_time, end_time, E_reqs, P_max, T_arr, T_dep, time_step, injection_price,consumption_price, pv_generation, L_c, N_ev, IDs, Stations, StationVector):
        # Define your environment-specific initialization here
        # Set up the necessary variables, observation space, and action space
        self.time_step = time_step
        self.end_time = end_time
        self.start_time= start_time
        self.day = 0
        self.P_max = P_max
        self.T_arr = T_arr
        self.T_dep = T_dep
        self.E_req = 0
        self.E_reqs = E_reqs
        self.L_c = L_c
        self.N_ev = N_ev
        self.StationVector = StationVector
        self.Stations = Stations
        self.IDs = IDs
        # Functions return the injection/consumption price at a time t
        self.lamda_inj= injection_price
        self.lamda_cons = consumption_price
        
        self.T = int(24*60/L_c)
        #Function returns the pv generation at a time t
        self.pv_generation = pv_generation
        
    # Define the objective function
    def objective_function(self, P_r):
        return -sum(self.C_elec(t, P_r) for t in range(self.T))

    # Define the constraint functions
    def constraint1(self, P_r):
        return sum(self.C_cons(t, P_r) for t in range(self.T)) - self.E_req
    
    def constraint2(self, P_r):
        return np.sum(P_r)* (self.L_c/ 60) - self.E_req
    
   #def constraint3(self, P_r):
       #return [P_r[i] - self.P_lim[i] for i in range(self.N_ev)]

    # Define the cost functions
    def C_elec(self, t, P_r):
        time = 24*self.day+self.start_time+t*self.L_c/60
        P_diff_t = np.sum(P_r[t*self.N_ev:(t+1)*self.N_ev])-self.pv_generation(time)
        if P_diff_t >= 0:
            return P_diff_t*(self.L_c/ 60)*self.lamda_cons(time)
        else:
            return P_diff_t*(self.L_c/ 60)*self.lamda_inj(time)
    
    def C_cons(self, t, P_r):
        time = 24*self.day+self.start_time+t*self.L_c/60
        P_diff_t = np.sum(P_r[t*self.N_ev:(t+1)*self.N_ev])-self.pv_generation(time)
        return P_diff_t * (self.L_c/ 60) * self.lamda_cons(time)
    
    def optimize(self):
        cost = 0
        while self.day * 24 < self.end_time:
            P_lim = self.get_P_lim_matrix()
            self.E_req = self.get_E_req()
            # Define the optimization problem
            P_r_initial_guess = P_lim
            constraints = [
                {'type': 'eq', 'fun': lambda P_r: self.constraint1(P_r)},
                {'type': 'eq', 'fun': lambda P_r: self.constraint2(P_r)}
            ]
            bounds = np.array( [(0, P_lim[i]) for i in range(len(P_lim))])
            result = minimize(self.objective_function, P_r_initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, jac = '2-point')
    
            cost += result.fun
            self.day += 1
        return cost
    
    def get_P_lim_matrix(self):
        # Arrival and departure time constraints are also introduced by this
        P_lim = np.zeros(self.N_ev*self.T)
        for i in range(self.T):
            time = self.day*24+self.start_time+i*self.L_c/60
            [_, _, P_limvec, _, _] = get_EV_fleet(time, self.IDs,self.E_req,self.P_max, self.T_arr, self.T_dep, self.Stations, self.StationVector, self.L_c)
            index = i*self.N_ev
            P_lim[index:index+self.N_ev] = P_limvec
        return P_lim
    
    def get_E_req(self):
        time = (self.day+1)*24 + self.start_time
        vec = np.where(np.ones(len(self.T_arr))*time >= self.T_arr, np.ones(len(self.T_arr)), np.zeros(len(self.T_arr)))
        vec = np.where(np.ones(len(self.T_arr))*(time-24) <= self.T_dep, vec, np.zeros(len(self.T_arr)))
        return np.sum(np.multiply(self.E_reqs, vec))
    
    
def get_forecast_measurement(file_path):
    # Read data from the Excel file
    df = pd.read_excel(file_path)
    data = df.values
    forecast = ((data[:, 2])[3:]/20)
    measurement = ((data[:, 4])[3:]/20)
    return np.append(forecast, 0.0), np.append(measurement, 0.0)

def get_price_forecast(file_path):
    # Read data from the Excel file
    df = pd.read_excel(file_path)
    data = df.values
    prices = data[:, 1]
    prices = np.array(prices, dtype=float)
    mask = ~np.isnan(prices)
    prices = prices[mask]
    return np.append(prices, 0.0)

def simulate_network_and_track(agent, env, num_episodes):
    costs = []
    demand_over_week = []
    cumulative_cost = 0
    price_rate = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_cost = 0

        while not done:
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_cost -= reward
            state = next_state

        costs.append(episode_cost)
        cumulative_cost += episode_cost
        demand_over_week.append(env.demand)  # Assuming your environment has a 'demand' attribute for weekly demand
        price_rate.append(env.price)        # Assuming your environment has a 'price' attribute for price rate

    return costs, demand_over_week, cumulative_cost, price_rate