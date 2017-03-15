#!/bin/bash
echo 'ysplit social cognitive predicition (bottom 30% vs top 70%) using resting state and xbrain correlations (resting state)'
./bin/xbrain \
    --connectivity='ts_rst' \
    --predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,scog_er40_cr_columnpcr_value' \
    --pct-variance=0.5 \
    --y-cutoff=0.3 \
    --method='ysplit' \
    --debug \
    --diagnostics \
    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

