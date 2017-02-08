#!/bin/bash
echo 'basic diagnosis prediction'
./bin/xbrain \
    --xcorr='ts_imi_resid,ts_obs_resid' \
    --connectivity='ts_rst' \
    --predict='Diagnosis' \
    --target-cutoff=0.5 \
    --pct-variance=0.8 \
    --target-group=0 \
    --debug \
    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

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
