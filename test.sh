#!/bin/bash
#echo 'hc/scz diagnosis prediction using resting state and xbrain correlations (IM/OB), all sites, 50% PCA variance retained'
#./bin/xbrain \
#    --xcorr='ts_imi_resid,ts_obs_resid' \
#    --connectivity='ts_rst' \
#    --predict='Diagnosis' \
#    --pct-variance=0.5 \
#    --target-group=0 \
#    --debug \
#    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

echo 'multiclass diagnosis prediction using resting state, CMH, 80% PCA variance retained'
./bin/xbrain \
    --connectivity='ts_rst' \
    --predict='Diagnosis' \
    --pct-variance=0.80 \
    --method='target' \
    --target-group=1 \
    --debug \
    --output='output/rest-diagnose' \
    /projects/jviviano/data/xbrain/assets/database_xbrain_CMH.csv

#echo 'ysplit social cognitive predicition (bottom 30% vs top 70%) using resting state and xbrain correlations (IM/OB), all sites, 50% PCA variance retained'
#./bin/xbrain \
#    --connectivity='ts_rst' \
#    --predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,scog_er40_cr_columnpcr_value' \
#    --pct-variance=0.5 \
#    --y-cutoff=0.3 \
#    --method='ysplit' \
#    --debug \
#    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

#echo 'ysplit social cognitive predicition (AUTO) using resting state and xbrain correlations (IM/OB), all sites, 50% PCA variance retained'
##./bin/xbrain \
#    --xcorr='ts_imi_resid,ts_obs_resid' \
#    --connectivity='ts_rst' \
#    --predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,scog_er40_cr_columnpcr_value' \
#    --pct-variance=0.5 \
#    --y-cutoff='auto' \
#    --method='ysplit' \
#    --debug \
#    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

#echo 'outlier detection using resting state, all sites 50% PCA variance retained'
#./bin/xbrain \
#    --xcorr='ts_imi_resid,ts_obs_resid' \
#    --connectivity='ts_rst' \
#    --predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,scog_er40_cr_columnpcr_value' \
#    --method='anomaly' \
#    --pct-variance=0.8 \
#    --debug \
#    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

#echo 'diagnosis classification using dynamic functional connectivity analysis, all spins data'
#./bin/xbrain \
#    --dynamics='ts_rst' \
#    --predict='Diagnosis' \
#    --method='target' \
#    --target-group=0 \
#    --debug \
#    /projects/jviviano/data/xbrain/assets/database_xbrain_SPN.csv

#echo 'diagnosis classification using dynamic functional connectivity analysis, all spins data'
#./bin/xbrain \
#    --dynamics='ts_ea1,ts_ea2,ts_ea3' \
#    --predict='Diagnosis' \
#    --method='target' \
#    --target-group=0 \
#    --debug \
#    /projects/jviviano/data/xbrain/assets/database_xbrain_SPN.csv
