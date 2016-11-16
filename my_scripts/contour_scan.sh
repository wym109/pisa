if [ "$HOSTNAME" = schwyz ]; then
    dev=0
    loop="2.35 2.8 2.4 2.85"
fi
if [ "$HOSTNAME" = uri ]; then
    dev=0
    loop="2.5 2.55 2.6 2.65"
fi
if [ "$HOSTNAME" = unterwalden ]; then
    dev=0
    loop="2.7 2.75 2.45 2.3"
fi
for delta in $loop
do
    echo ${delta}
    export CUDA_VISIBLE_DEVICES=$dev
    nohup python nutau/analysis.py \
    -t pisa/resources/settings/pipeline/numu_pingu_mc_v39_baseline.cfg \
    -m pisa/resources/settings/minimizer/slsqp_settings_fac1e9_eps1e-4_mi20.json \
    -pd asimov \
    --metric llh \
    --mode scan \
    --var theta23 \
    --range "np.arcsin(np.sqrt(np.linspace(0.3,0.7,21)))/np.pi*180*ureg.degree" \
    -spf "deltam31=${delta}e-3*units.eV**2" \
    -o my_results/pingu_v39/3_year_contours_newsysfits/2d_contour_t2k_baseline/deltam31_${delta}.json \
    2>&1 > /dev/null &
    ((dev++))
done
