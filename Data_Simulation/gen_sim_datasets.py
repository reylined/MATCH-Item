import sys
sys.path.append("C:/Users/reyli/Documents/GitHub/MATCH-Item")
from Data_Simulation.data_simulation_md import simJM_item_MD
import numpy as np

n_sim = 100           # For simulation study
n_sim_ablation = 100  # For ablation study
n_sim_GS = 10         # For grid search
I = 1000
num_items = np.array([23,10])
obstime = np.array([0,1,2,3,4,5,6,7,8,9,10])

# For Simulation Study
for i_sim in range(n_sim):
    np.random.seed(i_sim)
    
    data = simJM_item_MD(I, num_items, obstime)
    
    path = "G:/My Drive/Biostatistics/Dissertation/Item_Level/Simulation/Sim_datasets/"
    data.to_csv(path+"sim_MD"+str(i_sim)+".csv", index=False)

    
# For Ablation Study
for i_sim in range(n_sim_ablation):
    np.random.seed(i_sim + 4321)
    
    data = simJM_item_MD(I, num_items, obstime)
    
    path = "G:/My Drive/Biostatistics/Dissertation/Item_Level/Simulation/Sim_datasets/"
    data.to_csv(path+"sim_MD_ablation"+str(i_sim)+".csv", index=False)


# For Grid Search
for i_sim in range(n_sim_GS):
    np.random.seed(i_sim + 1234)
    
    data = simJM_item_MD(I, num_items, obstime)
    
    path = "G:/My Drive/Biostatistics/Dissertation/Item_Level/Simulation/Sim_datasets/"
    data.to_csv(path+"sim_MD_GS"+str(i_sim)+".csv", index=False)
    