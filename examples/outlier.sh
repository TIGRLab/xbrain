#!/bin/bash
echo 'outlier detection using resting state, all sites 50% PCA variance retained'
./bin/xbrain \
    --xcorr='ts_imi_resid,ts_obs_resid' \
    --connectivity='ts_rst' \
    --predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,scog_er40_cr_columnpcr_value' \
    --method='anomaly' \
    --pct-variance=0.8 \
    --debug \
    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

