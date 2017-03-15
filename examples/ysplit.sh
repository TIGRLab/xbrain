#!/bin/bash
echo 'ysplit social cognitive predicition (AUTO) using resting state and xbrain correlations (IM/OB), all sites, 50% PCA variance retained'
./bin/xbrain \
    --xcorr='ts_imi_resid,ts_obs_resid' \
    --connectivity='ts_rst' \
    --predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,scog_er40_cr_columnpcr_value' \
    --pct-variance=0.5 \
    --y-cutoff='auto' \
    --method='ysplit' \
    --debug \
    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

