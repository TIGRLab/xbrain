#!/bin/bash
echo 'hc/scz diagnosis prediction using resting state and xbrain correlations (IM/OB), all sites, 50% PCA variance retained'
./bin/xbrain \
    --xcorr='ts_imi_resid,ts_obs_resid' \
    --connectivity='ts_rst' \
    --predict='Diagnosis' \
    --pct-variance=0.5 \
    --method='target' \
    --target-group=0 \
    --debug \
    --output='output/imob-rest-diagnose' \
    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

