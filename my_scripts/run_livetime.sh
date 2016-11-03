dev=0
if [ "$HOSTNAME" = schwyz ]; then
    loop="1 2 4"
fi
if [ "$HOSTNAME" = uri ]; then
    loop="8 12 18 24"
fi
if [ "$HOSTNAME" = unterwalden ]; then
    loop="30 36 48 6"
fi
for y in $loop
do
    export CUDA_VISIBLE_DEVICES=$dev
    nohup python nutau/analysis.py \
    -t pisa/resources/settings/pipeline/nutau_pingu_mc_v47_optimistic.cfg \
    -m pisa/resources/settings/minimizer/slsqp_settings_fac1e9_eps1e-4_mi20.json \
    -pd asimov \
    --metric llh \
    --mode scan \
    --var nutau_norm \
    --range "np.array([0., 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0])*ureg.dimensionless" \
    -sp livetime=$y*units.months \
    -o my_results/pingu_v47/livetime_llh_new_optimistic/${y}_months_livetime.json \
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
