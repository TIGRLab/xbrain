#!/bin/bash
echo 'biotyping using resting state'
xbrain \
    --connectivity='ts_rst' \
    --biotype='ts_rst' \
    --predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,scog_er40_cr_columnpcr_value' \
    --pct-variance=0.8 \
    --y-cutoff=0.3 \
    --method='biotype' \
    --output='/projects/jviviano/code/xbrain/output/biotype' \
    --debug \
    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

