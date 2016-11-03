export CUDA_VISIBLE_DEVICES=$1
nohup python nutau/analysis.py \
-t pisa/resources/settings/pipeline/$2.cfg \
-m pisa/resources/settings/minimizer/slsqp_settings_fac1e9_eps1e-4_mi20.json \
-pd asimov \
--metric llh \
--mode scan \
--var nutau_norm \
--range "[0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]*ureg.dimensionless" \
-o my_results/pingu_v47/nutau_norm_new/$2.json \
2>&1 > /dev/null &
