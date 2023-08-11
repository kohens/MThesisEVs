import numpy as np
import PPO_update
import matplotlib.pyplot as plt
import pandas as pd
import gym
import matplotlib.pyplot as plt



#Choose number of charging stations
N_evList = np.array([2, 8, 16, 32, 64])

# Length of control time step in minutes
L_cList = np.array([5, 10, 15, 30, 45, 60])

# Number of past and future hours to consider in pv generation
N_past = 2
N_fut = 2



# Grid injection price: 1.46eu/MWh
lamda_inj = 1.46

# Step 1: Define the environment and other necessary components

kappaList = np.array([1, 1.5, 5, 10])
kappaList = 1.5*np.ones(1)
action_dim = 1
dim_Z_t = 2
dim_t = 1
dim_P_pv_state = N_past + 1
dim_P_f_pv_state = N_past + N_fut + 1
dim_L_cons = 1 + N_fut + 1
state_dim = dim_Z_t + dim_t + dim_P_pv_state + dim_P_f_pv_state + dim_L_cons


# Step 3: Set hyperparameters
epochs = 500
batch_size = 1
learning_rate = 0.001
gamma = 0.99
clip_epsilon = 0.2
value_coeff = 0.5
entropy_coeff = 0.01
lamda = 0.9

#Creation of the data
N_points = 1000

#Length of simulated episode in days
N_days = 7




# Arrival times 
# Normal distribution
mean_arrival = 8.5
std_arrival = 1
arrivals = np.random.normal(mean_arrival, std_arrival, N_points*epochs)


# Departure times 
# Normal distribution
mean_departure = 17.5
std_departure = 1.5
evening_departures = np.random.normal(mean_departure, std_departure, N_points*epochs)

mean_departure = 12.5
std_departure = 0.5
noon_departures = np.random.normal(mean_departure, std_departure, N_points*epochs)

a = np.random.uniform(0, 1, N_points*epochs)
Noon_probability = 1/8
departures = np.where(a >= Noon_probability, evening_departures, noon_departures)


# Number of stations used on a day: compound binomial distribution
# Probability a single loading station is used on a day 
P_used = 0.95

file_path = 'SolarForecast_20230501-20230507.xls'

end = N_days*24
end_time = end
[forecast, measurement] = PPO_update.get_forecast_measurement(file_path)
times = np.arange(0, len(forecast)*0.25, 0.25)
forecast = PPO_update.create_time_function(times, forecast)
pv_generation = PPO_update.create_time_function(times, measurement)

file_path = 'Day_aheadprices.xlsx'
lamda_belpex_forecast = PPO_update.get_price_forecast(file_path)

price_forecast = 1.21*(45*np.ones(len(lamda_belpex_forecast))+lamda_belpex_forecast)
times = np.arange(0, len(price_forecast), 1)
price_forecast = PPO_update.create_time_function(times, price_forecast)
lamda_belpex = lamda_belpex_forecast

consumption_prices = 1.21*(45*np.ones(len(times))+lamda_belpex)

consumption_price = PPO_update.create_time_function(times, consumption_prices)
injection_price = PPO_update.create_time_function(times, np.ones(len(times))*lamda_inj)
Stations_used = np.random.binomial(1, P_used, int(N_points*epochs*2/0.95))
for N_ev in N_evList:
    Station_nrs = np.arange(1, N_ev+1)
    Days = np.arange(1, int(len(Stations_used)/N_ev)+2)
    Station_nrs = np.tile(Station_nrs, int(len(Stations_used)/N_ev)+1)
    Days = np.repeat(Days, N_ev)
    Station_nrs = Station_nrs[0:len(Stations_used)]
    Days = Days[0:len(Stations_used)]
    Load_Stations = np.multiply(Station_nrs, Stations_used)
    Load_Stations = (Load_Stations[Load_Stations!=0])[0:len(arrivals)]
    Days = np.multiply(Days, Stations_used)
    Days = (Days[Days != 0])[0:len(arrivals)]
    StationVector = np.arange(0, N_ev)

    
    # Arrival and departure times starting from 0 until the end
    T_arr = arrivals + (Days-np.ones(len(Days)))*24
    T_dep = departures + (Days-np.ones(len(Days)))*24
    T_arr = np.sort(T_arr)

    # Remove invalid points
    check = T_arr < T_dep
    T_arr = T_arr[check]
    T_dep = T_dep[check]
    check = T_arr < 24*N_days*epochs
    T_arr = T_arr[check]
    T_dep = T_dep[check]
    start_time = 2

    Num_sessions = len(T_arr)

    # IDs of the load sessions: give each session a number
    IDs = np.arange(1, Num_sessions+1)

    Load_Stations = Load_Stations[0:Num_sessions]

    # Required energy: Average of 50 MWh with standard deviation of 20 MWh
    mean_capacity = 10 # 50kWh
    std_capacity = 5 # 20 kWh
    E_reqs = np.random.normal(mean_capacity, std_capacity, Num_sessions)
    # All reqs should be > 0
    E_reqs = np.where(E_reqs < 0, mean_capacity*np.ones(Num_sessions)-E_reqs, E_reqs)


    P_max = 11*np.ones(Num_sessions) # 11 kW
    for L_c in L_cList:
        moments = np.arange(start_time, end+1+L_c/60, L_c/60)
        forecast_list = forecast(moments)
        price_forecast_list = price_forecast(moments)
        for kappa in kappaList:
            # Create environment and train the network
            file_name = 'N_ev_'+str(N_ev)+'_L_c_'+str(L_c)+'_kappa_'+str(kappa)+'.npz'
            np.savez(file_name,array1=T_arr, array2=T_dep, array3=E_reqs, array4=forecast_list, array5=price_forecast_list, array6=moments, array7=pv_generation(moments), array8=consumption_price(moments))
            env = PPO_update.Environment(start_time, int(end_time+1), N_ev, forecast, price_forecast, N_past, N_fut, L_c, E_reqs, P_max, T_arr, T_dep, Load_Stations, StationVector, IDs, kappa ,injection_price,consumption_price, pv_generation )
            agent = PPO_update.PPOAgent(state_dim, action_dim, env, epochs, batch_size, clip_epsilon, value_coeff, entropy_coeff, clip_epsilon, learning_rate , gamma, lamda)
            PL, VL, R, A, S = agent.train()
            file_name = 'Results'+ file_name
            np.savez(file_name, policy_loss=PL, value_loss=VL, rewards=R, actions=A, states=S)
