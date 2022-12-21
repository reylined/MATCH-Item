import sys
sys.path.append("G:/My Drive/Biostatistics/Dissertation")
from Item_Level.Simulation.data_simulation_md import simJM_item_MD
#from Item_Level.Simulation.data_simulation_items import simJM_item

import numpy as np
import pandas as pd

'''
item_params = pd.read_csv("G:/My Drive/Biostatistics/Dissertation/Item_Level/Simulation/item_param_md.csv")
b = item_params["b"].values
a = item_params.loc[:, item_params.columns.str.startswith('a')].values
'''


n_sim = 100
I = 1000
num_items = np.array([23,10])
obstime = np.array([0,1,2,3,4,5,6,7,8,9,10])


for i_sim in range(n_sim):
    np.random.seed(i_sim)
    
    data_all = simJM_item_MD(I, num_items, obstime)
    #data = data_all[data_all.obstime < data_all.time]

    path = "G:/My Drive/Biostatistics/Dissertation/Item_Level/Simulation/Sim_datasets/"
    data_all.to_csv(path+"sim_MD"+str(i_sim)+".csv", index=False)