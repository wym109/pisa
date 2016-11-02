dev=0
if [ "$HOSTNAME" = schwyz ]; then
    loop="35 37 39"
fi
if [ "$HOSTNAME" = uri ]; then
    loop="41 43 45 47"
fi
if [ "$HOSTNAME" = unterwalden ]; then
    loop="49 51 53 55"
fi
for theta in $loop
do
    export CUDA_VISIBLE_DEVICES=$dev
    nohup python nutau/analysis.py \
    -t pisa/resources/settings/pipeline/numu_pingu_mc_v47_better_sys_res.cfg \
    -m pisa/resources/settings/minimizer/slsqp_settings_fac1e9_eps1e-4_mi20.json \
    -pd asimov \
    --metric llh \
    --mode scan \
    --var deltam31 \
    --range "np.linspace(2.0e-3,3.0e-3,11)*ureg.eV**2" \
    -spf "theta23=${theta}*units.degree" \
    -o my_results/pingu_v47/1_year_contours/2d_contour_t2k_better_sys_res/theta23_${theta}.json \
    2>&1 > /dev/null &
    ((dev++))
done
