import torch
import torch.optim as optim
import copy

import sys
sys.path.append("/content/gdrive/MyDrive/Colab Notebooks/MATCH-Item")
#sys.path.append("C:/Users/reyli/Documents/GitHub/MATCH-Item")
from MATCH_Models.MATCH_item.MATCH import MATCH
from MATCH_Models.MATCH_item.functions import (get_tensors, augment, format_output,
                                    CE_loss, ordinalOHE, init_weights)
from MATCH_Models.metrics import (AUC, Brier)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

import pickle

#seed = int(sys.argv[1])

# Ablation study settings
# One-hot-encoding
ordinal_enc = False
positive_constraint = "none"

# Simulation setting
I = 1000
num_items = [23,10]
K = sum(num_items)
obstime = np.array([0,1,2,3,4,5,6,7,8,9,10])
landmark_times = [2,3,4,5]
pred_windows = [1,2]

# Model setting
n_epoch = 30
batch_size = 32


n_sim = 100
AUC_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
iAUC_array = np.zeros((n_sim, len(landmark_times)))
true_AUC_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
true_iAUC_array = np.zeros((n_sim, len(landmark_times)))

BS_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
iBS_array = np.zeros((n_sim, len(landmark_times)))
true_BS_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
true_iBS_array = np.zeros((n_sim, len(landmark_times)))

train_loss_values = np.full((n_sim,n_epoch), np.nan)
test_loss_values = np.full((n_sim,n_epoch), np.nan)
best_epochs = np.full(n_sim, np.nan)




for i_sim in range(n_sim):
    
    print("i_sim:",i_sim)
    np.random.seed(i_sim)
    
    path = "/content/gdrive/MyDrive/Biostatistics/Dissertation/Item_Level/Simulation/Sim_datasets/"
    data_all = pd.read_csv(path+"sim_MD"+str(i_sim)+".csv")
    
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
    train_long = ordinalOHE(train_long.long(), ordinal=ordinal_enc, n_cat=4).permute(0,1,3,2)
    train_long, train_mask, train_e, train_t, subjid = augment(
                    train_long, None, train_mask, train_e, train_t, n_cat=4)
    
   
    test_long, test_mask, test_e, test_t, obs_time = get_tensors(df=test_data.copy(),
                                                                     long=item_vars,
                                                                     base=base_vars,
                                                                     obstime="obstime",
                                                                     roundnum=1)
    test_long = ordinalOHE(test_long.long(), ordinal=ordinal_enc, n_cat=4).permute(0,1,3,2)
    test_long, test_mask, test_e, test_t, test_subjid = augment(
                    test_long, None, test_mask, test_e, test_t, n_cat=4)
    
    
    # Train model
    
    torch.manual_seed(i_sim)
    model = MATCH(n_items = fixed_param["n_items"],
                  n_cat = fixed_param["n_cat"],
                  n_base = fixed_param["n_base"],
                  out_len = fixed_param["out_len"],
                  pos_constraint = positive_constraint)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    

    lowest_test_loss = float('inf')
    
    for epoch in range(n_epoch):
        running_loss = 0
        train_id_ = torch.from_numpy(np.random.permutation(train_id))
        model = model.train()
        for batch in range(0, len(train_id_), batch_size):
            optimizer.zero_grad()
            
            batch_id = train_id_[batch:batch+batch_size]
            
            indices = (subjid[..., None] == batch_id).any(-1).nonzero().squeeze() # indices of subjid in batch_id
            batch_long = train_long[indices,:,:,:]
            batch_mask = train_mask[indices,:,:]
            batch_t = train_t[indices]
            batch_e = train_e[indices]
            
            if len(indices)>1: #drop if last batch size is 1
                yhat_surv = torch.softmax(model(batch_long.float(), None, batch_mask), dim=1)
                s_filter, e_filter = format_output(obs_time, batch_mask, batch_t, batch_e, fixed_param["out_len"])
                loss = CE_loss(yhat_surv, s_filter, e_filter)
                loss.backward()
                optimizer.step() 
                running_loss += loss
                
        train_loss_values[i_sim,epoch] = running_loss.detach().numpy()
        
        
        # test loss
        model = model.eval()
        with torch.no_grad():
            surv_pred_test = torch.softmax(model(test_long.float(), None, test_mask), dim=1)
            s_filter_test, e_filter_test = format_output(obs_time, test_mask, test_t, test_e, fixed_param["out_len"])
            test_loss = CE_loss(surv_pred_test, s_filter_test, e_filter_test)
        test_loss_values[i_sim,epoch] = test_loss.detach().numpy()
        
        if test_loss < lowest_test_loss:
            lowest_test_loss = test_loss
            best_epochs[i_sim] = epoch
            best_model = copy.deepcopy(model.state_dict())
        
    #plt.plot(train_loss_values[i_sim])
    plt.plot(test_loss_values[i_sim])
    
    
    
    # Test model
    
    #model.load_state_dict(best_model)
    model = model.eval()
    
    for LT_index, LT in enumerate(landmark_times):
        
        pred_times = [x+LT for x in pred_windows]
        
        # Only keep subjects with survival time > landmark time
        tmp_data = test_data.loc[test_data["time"]>LT,:]
        tmp_id = np.unique(tmp_data["id"].values)
        tmp_all = data_all.loc[data_all["id"].isin(tmp_id),:]
    
        
        # Only keep longitudinal observations <= landmark time
        tmp_data = tmp_data.loc[tmp_data["obstime"]<=LT,:]
        true_prob_tmp = tmp_all.loc[tmp_all["obstime"].isin(pred_times), ["true"]].values.reshape(-1,len(pred_times))
        true_prob_LT = tmp_all.loc[tmp_all["obstime"]==LT, ["true"]].values
        true_prob_tmp = true_prob_tmp / true_prob_LT
        tmp_long, tmp_mask, e_tmp, t_tmp, obs_time = get_tensors(df=tmp_data.copy(),
                                                                           long=item_vars,
                                                                           base=base_vars,
                                                                           obstime="obstime",
                                                                           roundnum=1)
        tmp_long = ordinalOHE(tmp_long.long(), ordinal=ordinal_enc, n_cat=4).permute(0,1,3,2)
        
        with torch.no_grad():
            surv_pred = torch.softmax(model(tmp_long.float(), None, tmp_mask), dim=1)
            surv_pred = surv_pred.detach().numpy()
            surv_pred = surv_pred[:,::-1].cumsum(axis=1)[:,::-1]
            surv_pred = surv_pred[:,1:(fixed_param["out_len"]+1)]
    
        auc, iauc = AUC(surv_pred, e_tmp.numpy(), t_tmp.numpy(), np.array(pred_times))
        AUC_array[i_sim, LT_index, :] = auc
        iAUC_array[i_sim, LT_index] = iauc
        auc, iauc = AUC(true_prob_tmp, np.array(e_tmp), np.array(t_tmp), np.array(pred_times))
        true_AUC_array[i_sim, LT_index, :] = auc
        true_iAUC_array[i_sim, LT_index] = iauc

        bs, ibs = Brier(surv_pred, e_tmp.numpy(), t_tmp.numpy(),
                          train_e.numpy(), train_t.numpy(), LT, np.array(pred_windows))
        BS_array[i_sim, LT_index, :] = bs
        iBS_array[i_sim, LT_index] = ibs
        bs, ibs = Brier(true_prob_tmp, e_tmp.numpy(), t_tmp.numpy(),
                          train_e.numpy(), train_t.numpy(), LT, np.array(pred_windows))
        true_BS_array[i_sim, LT_index, :] = bs
        true_iBS_array[i_sim, LT_index] = ibs
    
    
    
np.set_printoptions(precision=3)
print("AUC:",np.nanmean(AUC_array, axis=0))
print("iAUC:",np.mean(iAUC_array, axis=0))
print("True AUC:",np.nanmean(true_AUC_array, axis=0))
print("True iAUC:",np.mean(true_iAUC_array, axis=0))

print("BS:\n", np.mean(BS_array, axis=0))
print("iBS:",np.mean(iBS_array, axis=0))
print("True BS:\n", np.mean(true_BS_array, axis=0))
print("True iBS:",np.mean(true_iBS_array, axis=0))
    

## save results
results = {"AUC":AUC_array,
           "iAUC":iAUC_array,
           "BS":BS_array,
           "iBS":iBS_array,
           "t_AUC":true_AUC_array,
           "t_iAUC":true_iAUC_array,
           "t_BS":true_BS_array,
           "t_iBS":true_iBS_array,
           "train_loss_val":train_loss_values,
           "test_loss_val":test_loss_values,
           "best_epochs":best_epochs
}


outfile = open('MATCH-Item/Simulation_Study/Results/MATCH_OHE-md.pickle', 'wb')
pickle.dump(results, outfile)
outfile.close() 
