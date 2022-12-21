library(dplyr)
library(pracma)
source("G:/My Drive/Biostatistics/Dissertation/NN/Models/AUC_BS.r")

setwd("G:/My Drive/Biostatistics/Dissertation/Item_Level/Simulation")


# Get names
simfiles = list.files(path="G:/My Drive/Biostatistics/Dissertation/Item_Level/Simulation/Sim_predictions/JM_grouped")
sim_nums = as.numeric(gsub(".*?([0-9]+).*", "\\1", simfiles))





# Multiple replications
n_sim = 100
landmark_times = c(2,3,4,5)
pred_windows = c(1,2)

AUC_array = array(NA, dim=c(n_sim, length(landmark_times), length(pred_windows)))
BS_array = array(NA, dim=c(n_sim, length(landmark_times), length(pred_windows)))


for(i_sim in 1:n_sim){
  #if(!((i_sim-1) %in% sim_nums)){next}
  
  dat = read.csv(paste0("Sim_datasets/sim_MD",i_sim-1,".csv"))
  dat$event = ifelse(dat$event=="True", TRUE, FALSE)
  dat_bl_train = dat %>%
    select(id, obstime, event, time) %>%
    filter(obstime==0) %>%
    filter(id < 0.7*length(unique(dat$id)))
  
  ### Pick for sumscore or grouped or grouped_wrong
  #filename = paste0("Sim_predictions/JM_sumscore/JM_sumscore_",i_sim-1,".rdata")
  #filename = paste0("Sim_predictions/JM_grouped/JM_grouped",i_sim-1,".rdata")
  filename = paste0("Sim_predictions/JM_grouped/JM_grouped_w",i_sim-1,".rdata")
  load(filename)
  
  for(iLT in 1:length(landmark_times)){
    LT = landmark_times[iLT]
    pred_times = LT + pred_windows
    
    ee = LT_event[[iLT]]
    tt = LT_time[[iLT]]
    ss = surv_pred_all[[iLT]]
    
    AUC_array[i_sim, iLT,] = AUC(ss, ee, tt, pred_times)$auc
    BS_array[i_sim, iLT,] = Brier(ss, ee, tt, dat_bl_train$event, dat_bl_train$time, LT, pred_windows)
  }
}




get_integrated = function(x, times){
  out = trapz(times,x) / (max(times) - min(times))
  return(out)
}


get_results = function(results, idt, times){
  results_deltaT = results[,,idt]
  out = round(apply(results_deltaT, c(2), mean, na.rm=T), digits=3)
  iout = apply(results_deltaT, c(1), function(x) get_integrated(x,times))
  iout = round(mean(iout, na.rm=T), digits=3)
  return(list(out,iout))
}

get_results(AUC_array,1,landmark_times+1)
get_results(AUC_array,2,landmark_times+2)
get_results(BS_array,1,landmark_times+1)
get_results(BS_array,2,landmark_times+2)

