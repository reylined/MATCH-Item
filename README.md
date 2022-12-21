# MATCH-Item
Implementation and simulation study for MATCH-Item: a convolutional neural network for item-level longitudinal and survival outcome data.

data simulation
- contains code to generate simulated datasets
	- [data_simulation_md.py] generates item-level longitudinal and survival data
	- [gen_sim_datasets.py] calls the sim_jm function from [data_simulation_md.py]
	  to create 100 simulated datasets

MATCH models
- contains Pytorch models and support functions for MATCH and MATCH-item models

simulation study
- contains code to run simulation study for each model
	-[JM_sumscore.R]     - corresponds to JM1 in simulation study
	-[JM_grouped.R]      - corresponds to JM2 and JM3 in simulation study
	-[MATCH_cont2_md.py] - corresponds to MATCH-net in simulation study
	-[MATCH_item_md.py]  - corresponds to MATCH-item in simulation study
	-[AUC_BS.R]          - support functions to calculate AUC and Brier score metrics
      -[AUC_BS_JM.R]       - calculate AUC and Brier score for JM models
	-[AUC_BS_MATCH.R]    - calculate AUC and Brier score for MATCH models
