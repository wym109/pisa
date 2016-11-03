if [ "$HOSTNAME" = schwyz ]; then
    dev=0
    loop="2.4 2.45 2.8"
fi
if [ "$HOSTNAME" = uri ]; then
    dev=0
    loop="2.5 2.55 2.6 2.65"
fi
if [ "$HOSTNAME" = unterwalden ]; then
    dev=2
    loop="2.7 2.75"
fi
for delta in $loop
do
    echo ${delta}
    export CUDA_VISIBLE_DEVICES=$dev
    nohup python nutau/analysis.py \
    -t pisa/resources/settings/pipeline/numu_pingu_mc_v47_baseline.cfg \
    -m pisa/resources/settings/minimizer/slsqp_settings_fac1e9_eps1e-4_mi20.json \
    -pd asimov \
    --metric llh \
    --mode scan \
    --var theta23 \
    --range "np.arcsin(np.sqrt(np.linspace(0.3,0.7,21)))/np.pi*180*ureg.degree" \
    -spf "deltam31=${delta}e-3*units.eV**2" \
    -o my_results/pingu_v47/3_year_contours_new/2d_contour_t2k_baseline/deltam31_${delta}.json \
    2>&1 > /dev/null &
    ((dev++))
done
