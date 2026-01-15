import torch
import torch.optim as optim

import sys
sys.path.append("/content/gdrive/MyDrive/Colab Notebooks")
#sys.path.append("C:/Users/reyli/Documents/GitHub/MATCH-Item")
from Item_Level.MATCH_Models.MATCH_item.MATCH import MATCH
from Item_Level.MATCH_Models.MATCH_item.functions import (get_tensors, augment, format_output,
                                    CE_loss, ordinalOHE, init_weights)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

from sklearn.model_selection import ParameterGrid
import pickle

I = 1000
num_items = [23,10]
K = sum(num_items)
obstime = np.array([0,1,2,3,4,5,6,7,8,9,10])

n_sim = 10

params = {
    "lconv": [16, 32],
    "lmask": [8, 16],
    "llin": [16, 32]
}
param_grid = ParameterGrid(params)


# If previous results exist, open. Otherwise create new dict for storing results across different simulation runs
try:
    outfile = open("Item_Level/Simulation/Results/param_loss.pickle", "rb+")
    param_loss = pickle.load(outfile)
except (OSError) as e:
    outfile = open('Item_Level/Simulation/Results/param_loss.pickle', 'wb')
    param_loss = {}
    for param_ in param_grid:
        param_loss.update({str(param_): np.full(n_sim, np.nan)})


def train_match(config, fixed_param, train_data_tensors):

    train_long = train_data_tensors["train_long"]
    train_mask = train_data_tensors["train_mask"]
    train_t = train_data_tensors["train_t"]
    train_e = train_data_tensors["train_e"]
    subjid = train_data_tensors["subjid"]
    
    model = MATCH(n_items = fixed_param["n_items"],
                  n_cat = fixed_param["n_cat"],
                  n_base = fixed_param["n_base"],
                  out_len = fixed_param["out_len"],
                  l1 = config["lconv"],
                  l23 = config["lconv"],
                  l4 = config["lconv"],
                  lmask = config["lmask"],
                  llin = config["llin"])
    model.apply(init_weights)
    model = model.train()
    optimizer = optim.Adam(model.parameters())
    
    n_epoch = 30
    batch_size = 32
    
    loss_values = []
    for epoch in range(n_epoch):
        running_loss = 0
        train_id = torch.from_numpy(np.random.permutation(train_data["id"]))
        for batch in range(0, len(train_id), batch_size):
            optimizer.zero_grad()
            
            batch_id = train_id[batch:batch+batch_size]
            
            indices = (subjid[..., None] == batch_id).any(-1).nonzero().squeeze() # indices of subjid in batch_id
            batch_long = train_long[indices,:,:,:]
            batch_mask = train_mask[indices,:,:]
            batch_t = train_t[indices]
            batch_e = train_e[indices]
            
            if indices.shape[0] > 1: #drop if last batch size is 1
                yhat_surv = torch.softmax(model(batch_long.float(), None, batch_mask), dim=1)
                s_filter, e_filter = format_output(obs_time, batch_mask, batch_t, batch_e, fixed_param["out_len"])
                loss = CE_loss(yhat_surv, s_filter, e_filter)
                loss.backward()
                optimizer.step() 
                running_loss += loss
        loss_values.append(running_loss.tolist())
        
    plt.plot(loss_values)
    
    
    return model



def get_test_loss(model, fixed_param, test_data_tensors):
    
    test_long = test_data_tensors["test_long"]
    test_mask = test_data_tensors["test_mask"]
    test_t = test_data_tensors["test_t"]
    test_e = test_data_tensors["test_e"]
    
    model = model.eval()
    surv_pred = torch.softmax(model(test_long.float(), None, test_mask), dim=1)
    s_filter, e_filter = format_output(obs_time, test_mask, test_t, test_e, fixed_param["out_len"])
    loss = CE_loss(surv_pred, s_filter, e_filter)
    
    return loss
    


# loop over n_sim
for i_sim in range(n_sim):
    
    print("i_sim:",i_sim)
    np.random.seed(i_sim)
    
    path = "/content/gdrive/MyDrive/Biostatistics/Dissertation/Item_Level/Simulation/Sim_datasets/"
    data_all = pd.read_csv(path+"sim_MD_GS"+str(i_sim)+".csv")
    
    # Only observations occuring before the event time should be used for training
    data = data_all[data_all.obstime < data_all.time]
    data_bl = data.loc[data.obstime==0,:]
    
    base_vars = []
    item_vars = [i for i in data.columns if i.startswith("item")]
    other_vars = ["id","event","time","obstime"]
    
    fixed_param = {
        "n_items": len(item_vars),
        "n_cat": 4,
        "n_base": len(base_vars),
        "out_len": 4
    }
    
    # split train/test
    random_id = range(I)
    train_id = random_id[0:int(0.7*I)]
    test_id = random_id[int(0.7*I):I]
    
    train_data = data[data["id"].isin(train_id)]
    test_data = data[data["id"].isin(test_id)]
    
    train_long, train_mask, train_e, train_t, obs_time = get_tensors(df=train_data.copy(),
                                                                     long=item_vars,
                                                                     base=base_vars,
                                                                     obstime="obstime",
                                                                     roundnum=1)
    train_long = ordinalOHE(train_long.long(), n_cat=4).permute(0,1,3,2)
    train_long, train_mask, train_e, train_t, subjid = augment(
                    train_long, None, train_mask, train_e, train_t, n_cat=4)
    
    train_data_tensors = {
        "train_long": train_long,
        "train_mask": train_mask,
        "train_e": train_e,
        "train_t": train_t,
        "subjid": subjid
    }
    
    test_long, test_mask, test_e, test_t, obs_time = get_tensors(df=test_data.copy(),
                                                                     long=item_vars,
                                                                     base=base_vars,
                                                                     obstime="obstime",
                                                                     roundnum=1)
    test_long = ordinalOHE(test_long.long(), n_cat=4).permute(0,1,3,2)
    test_long, test_mask, test_e, test_t, subjid = augment(
                    test_long, None, test_mask, test_e, test_t, n_cat=4)
    
    test_data_tensors = {
        "test_long": test_long,
        "test_mask": test_mask,
        "test_e": test_e,
        "test_t": test_t
    }
    
    # loop over parameters
    for param_ in param_grid:
        print(param_)
        
        model = train_match(param_, fixed_param, train_data_tensors)
        
        test_loss = get_test_loss(model, fixed_param, test_data_tensors)
        param_loss.get(str(param_))[i_sim] = test_loss.item()
    
    # checkpoint to save results
    pickle.dump(param_loss, outfile)
    print(param_loss)

# close the checkpoint results file
outfile.close() 
        




# average over simulations and print param combination with lowest loss
'''
param_loss_mean = {k:np.mean(v) for k,v in param_loss.items()}
print(param_loss_mean)

print(min(param_loss_mean, key = param_loss_mean.get))
'''
