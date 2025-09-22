import sys
sys.path.append("C:/Users/reyli/Documents/GitHub/MATCH-Item")
from Data_Simulation.data_simulation_md import simJM_item_MD
import numpy as np

n_sim = 100
I = 1000
num_items = np.array([23,10])
obstime = np.array([0,1,2,3,4,5,6,7,8,9,10])


for i_sim in range(n_sim):
    np.random.seed(i_sim)
    
    data = simJM_item_MD(I, num_items, obstime)
    
    path = "C:/Users/reyli/Documents/GitHub/MATCH-Item/Data_Simulation/Datasets/"
    data.to_csv(path+"sim_MD"+str(i_sim)+".csv", index=False)