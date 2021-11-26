Embryo implantation outcome prediction model

place embryo features in "EmbyroFeatures" folder


####
Run Load_TSC_train.py with the following the specified arguments to train different models.

###########################
    FCN               :        mode=1,   ffh=16
    FCN + GTA         :        mode=301, ffh=16
    FCN + TPS         :        mode=400, adaD=3
    
    ResNet            :        mode=2,   ffh=16
    ResNet + GTA      :        mode=122, ffh=16
    ResNet + TPS      :        mode=400, adaD=4
    
    Inception32       :        mode=1400, hids=32 ,   ffh=16
    Inception32 + GTA :        mode=1402, hids=32 ,   ffh=16
    Inception32 + TPS :        mode=1408, hids=32
    
    
    Inception64       :        mode=1400, hids=64 ,   ffh=16
    Inception64 + GTA :        mode=1402, hids=64 ,   ffh=16
    Inception64 + TPS :        mode=1408, hids=64 
    
    
    E+TPS             :        mode=400, adad=1
    E+TPS             :        mode=400, adad=1, LrMo=3
    
    TPS-------
    
    No PE             : pos_enc=0
    Learnable PE      : pos_enc=2
    Fixed Function PE : pos_enc=1

    Non TPS-----------
    pos_enc=0
    -------------
############################

set --day5 to 1 for day 5 and 0 for day 3


##############################
set --Ra between 0-4 for different folds as test set.
set --JobId to different numbers every run and average the results.

#############

run load_2p_tsc.py to produce majority vote results.

specify paths for "mod_path3" and "mod_path5" to the location of saved TSC models.
Specify --JobId3 and --JobId5 for different iteration of saved models. And set Ra for different folds.
