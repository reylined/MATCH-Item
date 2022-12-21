import pickle
import numpy as np
import pandas as pd




path = 'G:/My Drive/Biostatistics/Dissertation/Item_Level/Simulation/Sim_predictions/'


infile = open(path + 'MATCH_item.pickle', 'rb')
#infile = open(path + 'MATCH_cont2.pickle', 'rb')
results = pickle.load(infile)
infile.close


def get_integrated(x, times):
    return np.trapz(x,times) / (max(times)-min(times))

np.set_printoptions(precision=3)
np.mean(results['AUC'], axis=0)
np.mean(results['BS'], axis=0)

np.mean(get_integrated(results['AUC'][:,:,0], [2,3,4,5]))
np.mean(get_integrated(results['BS'][:,:,0], [2,3,4,5]))

np.mean(get_integrated(results['AUC'][:,:,1], [2,3,4,5]))
np.mean(get_integrated(results['BS'][:,:,1], [2,3,4,5]))
