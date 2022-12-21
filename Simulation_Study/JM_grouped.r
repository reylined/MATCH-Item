library(dplyr)
library(JMbayes)

args = commandArgs(trailingOnly = TRUE)
i_sim = as.numeric(args[1])
set.seed(i_sim) # use 0-based indexing

setwd("G:/My Drive/Biostatistics/Dissertation/Item_Level/Simulation")

dat = read.csv(paste0("Sim_datasets/sim_MD",i_sim,".csv"))

### Uncomment to get JM3 Table 1:
#dat$subscore1 = rowSums(dat[,c(7:24,35:39)]) # mix up subscore
#dat$subscore2 = rowSums(dat[,c(25:36)])
dat = dat %>% select(id, obstime, event, time, subscore1, subscore2)
dat$event = ifelse(dat$event=="True", TRUE, FALSE)

I = length(unique(dat$id))
landmark_times = c(2,3,4,5)
pred_windows = c(1,2)


# Split train/test set
train_data = dat %>% filter(id < 0.7*I)
test_data = dat %>% filter(id >= 0.7*I)

# Longitudinal Submodel
mvlmefit = mvglmer(list(subscore1 ~ obstime + (obstime | id),
                        subscore2 ~ obstime + (obstime | id)), data = train_data,
                   families = list(gaussian, gaussian))

# Survival Submodel
survfit = coxph(Surv(time,event) ~ 1, data=train_data%>%filter(obstime==0), model=TRUE)

# Joint Model
jmfit <- mvJointModelBayes(mvlmefit, survfit, timeVar = "obstime")


# Dynamic Prediction
LT_id = vector("list",length(landmark_times))
LT_time = vector("list",length(landmark_times))
LT_event = vector("list",length(landmark_times))
surv_pred_all = vector("list",length(landmark_times))

for(iLT in 1:length(landmark_times)){
  LT = landmark_times[iLT]
  pred_times = LT + pred_windows
  
  tmp_data = test_data %>%
    filter(time > LT) %>% # Only keep subjects with survival time > landmark time
    filter(obstime <= LT) %>% # Only keep longitudinal observations <= landmark time
    arrange(time, id, obstime) # sort by event-time
  
  survpred = JMbayes::survfitJM(jmfit, newdata=tmp_data, idVar="id", survTimes=pred_times)
  survprob = do.call(rbind, lapply(1:length(unique(tmp_data$id)), function(x) survpred[[1]][[x]]))[,2]
  survprob = matrix(survprob, ncol=2, byrow=TRUE)
  
  
  LT_id[[iLT]] = tmp_data[tmp_data$obstime==0,]$id
  LT_time[[iLT]] = tmp_data[tmp_data$obstime==0,]$time
  LT_event[[iLT]] = tmp_data[tmp_data$obstime==0,]$event
  surv_pred_all[[iLT]] = survprob
}

filename = paste0("Sim_predictions/JM_grouped/JM_grouped_w",i_sim,".rdata")
save(LT_id, LT_time, LT_event, surv_pred_all, file=filename)
