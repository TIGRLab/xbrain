#!/bin/bash
echo 'diagnosis classification using dynamic functional connectivity analysis (SPINS)'
xbrain \
    --dynamics='ts_rst' \
    --predict='Diagnosis' \
    --method='target' \
    --target-group=0 \
    --debug \
    /projects/jviviano/data/xbrain/assets/database_xbrain_SPN.csv

