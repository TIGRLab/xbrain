#!/bin/bash
#echo 'diagnosis prediction, all sites, 50% PCA variance retained'
#./bin/xbrain \
#    --xcorr='ts_imi_resid,ts_obs_resid' \
#    --connectivity='ts_rst' \
#    --predict='Diagnosis' \
#    --target-cutoff=0.5 \
#    --pct-variance=0.5 \
#    --target-group=0 \
#    --debug \
#    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

echo 'multiclass diagnosis prediction using resting state, CMH, 80% PCA variance retained'
./bin/xbrain \
    --xcorr='ts_imi_resid,ts_obs_resid' \
    --connectivity='ts_rst' \
    --predict='Diagnosis' \
    --pct-variance=0.80 \
    --target-group=0 \
    --method='multiclass' \
    --debug \
    /projects/jviviano/data/xbrain/assets/database_xbrain_CMH.csv

#./bin/xbrain \
#    --xcorr='ts_imi_resid,ts_obs_resid' \
#    --connectivity='ts_rst' \
#    --predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,scog_er40_cr_columnpcr_value' \
#    --pct-variance=0.8 \
#    --target-cutoff=0.3 \
#    --target-group=1 \
#    --debug \
#    /projects/jviviano/data/xbrain/assets/database_xbrain.csv
#
