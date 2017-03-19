#!/bin/bash
echo 'biotyping using resting state'
#--predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,rad_total,iri_factor_pt,iri_factor_fs,iri_factor_ec,iri_factor_pd,iri_total,scog_er40_cr_columnpcr_value,scog_er40_cr_columnpcr_value,scog_er40_crt_columnqcrt_value,scog_er40ang_columnv,scog_er40fear_columnw,scog_er40hap_columnx,scog_er40noe_columny,scog_er40sad_columnz,scog_er40_fpa_columnaa,scog_er40_fpf_columnab,scog_er40mild_columnap,scog_er40extr_columnaq,scog_er40mdrt_columnar,scog_er40exrt_columnas,bprs_factor_neg_symp,bprs_factor_pos_symp,bprs_factor_activation,bprs_factor_hostility,bprs_factor_total,np_domain_tscore_process_speed,np_domain_tscore_att_vigilance,np_domain_tscore_work_mem,np_domain_tscore_verbal_learning,np_domain_tscore_visual_learning,np_domain_tscore_reasoning_ps,np_domain_tscore_social_cog,np_composite_tscore' \
xbrain \
    --connectivity='ts_rst' \
    --xcorr='ts_imi,ts_obs' \
    --biotype='ts_rst' \
    --pct-variance=0.9 \
    --k=5 \
    --predict='Part1_TotalCorrect,Part2_TotalCorrect,Part3_TotalCorrect,RMET total' \
    --target-group=0 \
    --method='biotype' \
    --output='/projects/jviviano/code/xbrain/output/biotype' \
    /projects/jviviano/data/xbrain/assets/database_xbrain.csv

