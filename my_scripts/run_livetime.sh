dev=0
#for y in {1..4}
#for y in {5..8}
#for y in {9..11}
#for y in 2 4 6 8
#for y in 12 18 24 36
for y in 48 72 96
#for y in 12 24 36 48
#for y in 60 72 84 96
#for y in 108 120 132
#pingu
do
    export CUDA_VISIBLE_DEVICES=$dev
    nohup python nutau/analysis.py \
    -t pisa/resources/settings/pipeline/nutau_pingu_mc_v47.cfg \
    -m pisa/resources/settings/minimizer/slsqp_settings_fac1e9_eps1e-4_mi20.json \
    -pd asimov \
    --metric llh \
    --mode scan21 \
    -sp livetime=$y*units.months \
    --var nutau_norm \
    -o my_results/pingu_v47/livetime_llh_kde/${y}_months_livetime.json \
    2>&1 > /dev/null &
    ((dev++))
done
# Deepcore nutau
#do
#    export CUDA_VISIBLE_DEVICES=$dev
#    nohup python nutau/analysis.py \
#    -t pisa/resources/settings/pipeline/nutau_deepcore_mc.cfg \
#    -t pisa/resources/settings/pipeline/nutau_deepcore_icc.cfg \
#    -m pisa/resources/settings/minimizer/slsqp_settings_fac1e9_eps1e-4_mi20.json \
#    -pd asimov \
#    --metric llh \
#    --mode scan21 \
#    -sp livetime=$y*units.months \
#    --var nutau_norm \
#    -o deepcore_livetime/nutau_norm/${y}_months_livetime.json \
#    2>&1 > /dev/null &
#    ((dev++))
#done
##Deepcore
#do
#    export CUDA_VISIBLE_DEVICES=$dev
#    nohup python pisa/core/analysis.py \
#    -t pisa/resources/settings/pipeline_settings/pipeline_settings_nutau_mc_2020.ini \
#    -t pisa/resources/settings/pipeline_settings/pipeline_settings_nutau_icc_2020.ini \
#    -m pisa/resources/settings/minimizer_settings/bfgs_settings_fac1e9_eps1e-4_mi20.json \
#    -pd asimov \
#    --metric llh \
#    --mode scan21 \
#    -sp livetime=$y*units.common_year \
#    -o deepcore_livetime_scenario1/${y}_years_livetime.json \
#    2>&1 > /dev/null &
#    ((dev++))
#done
