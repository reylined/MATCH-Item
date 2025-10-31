import torch
import torch.optim as optim

import sys
#sys.path.append("/content/gdrive/MyDrive/Colab Notebooks")
sys.path.append("C:/Users/reyli/Documents/GitHub/MATCH-Item")
from MATCH_Models.MATCH_item.MATCH import MATCH
from MATCH_Models.MATCH_item.functions import (get_tensors, augment, format_output,
                                    CE_loss, ordinalOHE, init_weights)
from Simulation_Study.metrics import (AUC, Brier)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


import pickle

#seed = int(sys.argv[1])


I = 1000
num_items = [23,10]
K = sum(num_items)
obstime = np.array([0,1,2,3,4,5,6,7,8,9,10])
landmark_times = [2,3,4,5]
pred_windows = [1,2]


n_sim = 1
AUC_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
iAUC_array = np.zeros((n_sim, len(landmark_times)))
true_AUC_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
true_iAUC_array = np.zeros((n_sim, len(landmark_times)))

BS_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
iBS_array = np.zeros((n_sim, len(landmark_times)))
true_BS_array = np.zeros((n_sim, len(landmark_times), len(pred_windows)))
true_iBS_array = np.zeros((n_sim, len(landmark_times)))


for i_sim in range(n_sim):
    print("i_sim:",i_sim)
    np.random.seed(i_sim)
    
    path = "G:/My Drive/Biostatistics/Dissertation/Item_Level/Simulation/Sim_datasets/"
    data_all = pd.read_csv(path+"sim_MD"+str(i_sim)+".csv")
    data = data_all[data_all.obstime < data_all.time]
    data_bl = data.loc[data.obstime==0,:]
    
    base_vars = []
    item_vars = [i for i in data.columns if i.startswith("item")]
    other_vars = ["id","event","time","obstime"]
    
    ## split train/test
    random_id = range(I) #np.random.permutation(range(I))
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
    
    
    
    ## Train model
    torch.manual_seed(i_sim)
    
    out_len = 4
    model = MATCH(n_items = len(item_vars),
                  n_cat = 4,
                  n_base = len(base_vars),
                  out_len = out_len,
                  pos_constraint="none")
    model.apply(init_weights)
    model = model.train()
    optimizer = optim.Adam(model.parameters())
    
    n_epoch = 30
    batch_size = 32
    
    loss_values = []
    for epoch in range(n_epoch):
        running_loss = 0
        train_id = torch.from_numpy(np.random.permutation(train_id))
        for batch in range(0, len(train_id), batch_size):
            optimizer.zero_grad()
            
            batch_id = train_id[batch:batch+batch_size]
            
            indices = (subjid[..., None] == batch_id).any(-1).nonzero().squeeze() # indices of subjid in batch_id
            batch_long = train_long[indices,:,:,:]
            batch_mask = train_mask[indices,:,:]
            batch_t = train_t[indices]
            batch_e = train_e[indices]
            
            if len(indices)>1: #drop if last batch size is 1
                yhat_surv = torch.softmax(model(batch_long.float(), None, batch_mask), dim=1)
                s_filter, e_filter = format_output(obs_time, batch_mask, batch_t, batch_e, out_len)
                loss = CE_loss(yhat_surv, s_filter, e_filter)
                loss.backward()
                optimizer.step()
                '''
                for p in model.item.convolution[0].parameters():
                    p.data.clamp_(0)
                '''
                running_loss += loss
        loss_values.append(running_loss.tolist())
    plt.plot(loss_values)
    
    
    
    # Test model
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
        tmp_long = ordinalOHE(tmp_long.long(), n_cat=4).permute(0,1,3,2)
        
        model = model.eval()
        surv_pred = torch.softmax(model(tmp_long.float(), None, tmp_mask), dim=1)
        surv_pred = surv_pred.detach().numpy()
        surv_pred = surv_pred[:,::-1].cumsum(axis=1)[:,::-1]
        surv_pred = surv_pred[:,1:(out_len+1)]
    
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
           "t_iBS":true_iBS_array
}

'''
outfile = open('Item_Level/Simulation/Results/MATCH_item.pickle', 'wb')
pickle.dump(results, outfile)
outfile.close() 
'''