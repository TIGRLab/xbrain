#!/bin/bash
echo 'ysplit social cognitive predicition (bottom 30% vs top 70%) using resting state and xbrain correlations (resting state)'
xbrain \
    --xcorr='ts_imi_resid,ts_obs_resid' \
    --connectivity='ts_rst' \
    --predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,scog_er40_cr_columnpcr_value' \
    --pct-variance=0.8 \
    --y-cutoff=0.3 \
    --method='ysplit' \
    --output='/home/jviviano' \
    --debug \
    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

