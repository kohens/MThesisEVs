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

# Inputs to provide:
    # array with forecasted electricity price and forecasted pv generation
    # array with arrival and departure times of EVs (real values)
    # array with required energy and max power of each loading session
    # array with loading stations of each loading session
    



# Help functions



def av(Array, time1, time2):
    if time1 >= time2:
        return Array[time2]
    else: 
        return np.mean(Array[time1:time2])

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
    while remaining_power > 0 and np.sum(P_b) < np.sum(E_req):
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
        
        # 1: Calculation of new state
        # Updates that do not depend on the action
        self.t += self.L_c/60
        P_pv_past = self.get_past_PV()
        P_pv_current = self.get_and_update_PV()
        P_pv_state_t = np.append(P_pv_past,P_pv_current)
        P_f_pv_state_t = self.get_forecast_tuple()
        L_cons = self.get_price_tuple()

        # Update of the required energy of the EV-fleet
        [P_charge, P_min, P_max] = self.divide_aggregated_power(action)
        P_charge = backup_controller(P_charge, P_min, P_max)
        [M_start, M_end] = self.get_M_vectors()
        E_charged = np.multiply((M_end-M_start)/60, P_charge)
        self.E_req -= E_charged
        self.E_req = np.maximum(self.E_req, np.zeros(self.N_ev))
        P_charge = real_time_controller(P_charge, M_start, M_end, self.L_c, self.N_ev, self.P_lim, self.t, self.kappa, self.pv_generation)
        [self.E_req, self.P_lim, self.IDs_fleet, self.EV_levels, self.T_arr_fleet, self.T_dep_fleet] = self.update_fleet()
        Z_t = get_fleet_state(self.E_req, self.P_lim)
        state = np.concatenate((Z_t, (self.t % 24)*np.ones(1), P_pv_state_t, P_f_pv_state_t, L_cons))
        # 2: Calculation of the reward
        reward = self.calculate_reward(P_charge)
        
        # 3: Check if episode is done: if time = end_time
        done = False
        if self.t > self.end_time-self.L_c/60:
            done = True
            self.done_procedure()
        
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
        return np.concatenate((Z_t, self.t*np.ones(1), P_pv_state_t, P_f_pv_state_t, L_cons))

    
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
    
        index_rest = ((self.t - self.start_time) * 60 / self.L_c + self.N_fut * 60 / self.L_c + 1).astype(int)
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
        # Go to an new epoch, move the data of the next week to the front
        self.T_arr -= self.end_time*np.ones(len(self.T_arr))
        self.T_dep -= self.end_time*np.ones(len(self.T_dep))
        
# Step 2: Initialize the policy network

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_size, output_size):
       super(ActorCriticPolicy, self).__init__()
       self.shared_layer = nn.Linear(input_size, 128)
       
       self.actor_layer1 = nn.Linear(128, 64)
       self.actor_layer2 = nn.Linear(64, 64)
       self.actor_output = nn.Linear(64, output_size)
       
       self.critic_layer1 = nn.Linear(128, 64)
       self.critic_layer2 = nn.Linear(64, 64)
       self.critic_output = nn.Linear(64, 1)

    def forward(self, x):
       x =  torch.tanh(self.shared_layer(x))
       
       actor_x =  torch.tanh(self.actor_layer1(x))
       actor_x =  torch.tanh(self.actor_layer2(actor_x))
       actor =  torch.tanh(self.actor_output(actor_x))
       
       critic_x =  torch.tanh(self.critic_layer1(x))
       critic_x =  torch.tanh(self.critic_layer2(critic_x))
       critic =  torch.tanh(self.critic_output(critic_x))

       return actor, critic
   
    def denormalize_action(self, normalized_action, a_max):
        denormalized_action = (normalized_action + 1) * 0.5 * (a_max)
        return denormalized_action
    
    def denormalize_value(self, normalized_value):
        # Inverse tanh to denormalize the value
        denormalized_value = 1000*torch.atanh(normalized_value)
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
        self.action = action_dim
        self.network = ActorCriticPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
    def collect_data(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
    
        for i in range(self.batch_size):
            state = self.env.reset()
            done = False
            while not done:
                states.append(state)
                P_agg = state[1]
                action = self.select_action(state, P_agg)
                next_state, reward, done = self.env.step(action)
                actions.append(action)
                rewards.append(reward)  # Append the actual reward value
                next_states.append(next_state)
                dones.append(done)
                state = next_state
        
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
    
        return  torch.tensor(states, dtype=torch.float32),torch.tensor(actions, dtype=torch.float32),torch.tensor(rewards, dtype=torch.float32), torch.tensor(next_states, dtype=torch.float32),torch.tensor(dones, dtype=torch.float32)

    
    def select_action(self, state, action_max):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_tensor = self.network(state)[0]
        normalized_action = action_tensor.item()
        action = self.network.denormalize_action(normalized_action, action_max)
        return action

    def calculate_advantages_and_targets(self, rewards, next_states, dones):

        values = self.network(next_states)[1].detach().squeeze()
        denormalized_values = self.network.denormalize_value(values)
    
        # Compute the target values using the Bellman equation without iteration
        target_values = rewards + self.discount_factor * (1 - dones) * torch.roll(denormalized_values, -1)
        target_values[-1] = 0  # Set the last target value to 0 since there's no next state
    
        # Compute the advantages using Generalized Advantage Estimation without iteration
        delta = rewards + self.discount_factor * (1 - dones) * torch.roll(denormalized_values, -1) - values
    
        # Calculate the discount factor for GAE without iteration
        discount_factor_gae = self.discount_factor * self.lamda * (1 - dones)
    
        # Calculate the GAE with tensor operations
        gae = torch.zeros_like(rewards)
        gae[-1] = delta[-1]
        for i in reversed(range(len(rewards) - 1)):
            gae[i] = delta[i] + discount_factor_gae[i] * gae[i + 1]
        advantages = gae.float()
        
        print(advantages)
        print(target_values)
        return advantages, target_values

    def update_policy(self):
        states, actions, rewards, next_states, dones = self.collect_data()

        advantages, target_values = self.calculate_advantages_and_targets(rewards, next_states, dones)

        self.optimizer.zero_grad()
        actor_output, critic_output = self.network(states)

        # Denormalize the critic's output to bring it back to the original value range
        denormalized_critic_output = self.network.denormalize_value(critic_output)

        # Calculate the value loss for the critic
        value_loss = F.mse_loss(denormalized_critic_output.squeeze(), target_values)

        # Calculate the policy loss for the actor using clipped surrogate objective
        new_log_probs = actor_output
        old_log_probs = self.network(next_states)[0]
        log_ratios = new_log_probs - old_log_probs
        
        # Clip the log probability ratios
        clipped_ratios = torch.clamp(log_ratios.exp(), 1 - self.clip_epsilon, 1 + self.clip_epsilon)
    
        # Construct the clipped surrogate objective
        surr1 = log_ratios * advantages
        surr2 = clipped_ratios * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Calculate the entropy term
        entropy = self.calculate_entropy(states)

        # Calculate the total loss and backpropagate
        total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        total_loss.backward()
        self.optimizer.step()

        return policy_loss, value_loss, rewards, actions, states

    def train(self):
        # Initialize lists to store tensors for each epoch
        policy_losses = []
        value_losses = []
        rewards_list = []
        actions_list = []
        states_list = []
    
        for epoch in range(self.epochs):
            # Assuming that self.update_policy() returns torch tensors
            policy_loss, value_loss, rewards, actions, states = self.update_policy()
            # Make sure to detach the tensors to prevent backpropagation
            policy_loss = policy_loss.detach()
            value_loss = value_loss.detach()
            rewards = rewards.detach()
            actions = actions.detach()
            states = states.detach()
    
            # Append tensors to the corresponding lists
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            rewards_list.append(rewards)
            actions_list.append(actions)
            states_list.append(states)
    
        # Pad the lists to create tensors of the same length
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
    
        # Convert torch tensors to NumPy arrays
        PL = policy_losses
        VL = value_losses
        R = padded_rewards.numpy()
        A = padded_actions.numpy()
        S = padded_states.numpy()
    
        return PL, VL, R, A, S
            
    def calculate_cost(self, env):
        # Collect total charging plan as well to visualize
        cost = 0
        costs = []
        actions = []
        state = self.env.reset()
        done = False
        while not done:
            action = self.select_action(state)
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
    forecast = (data[:, 2])[3:]/20
    measurement = (data[:, 4])[3:]/20
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
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_cost -= reward
            state = next_state

        costs.append(episode_cost)
        cumulative_cost += episode_cost
        demand_over_week.append(env.demand)  # Assuming your environment has a 'demand' attribute for weekly demand
        price_rate.append(env.price)        # Assuming your environment has a 'price' attribute for price rate

    return costs, demand_over_week, cumulative_cost, price_rate







