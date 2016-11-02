python nutau/plot_contour.py \
--dir1 my_results/pingu_v47/3_year_contours/2d_contour_nova_better_sys_res \
--dir2 my_results/pingu_v47/3_year_contours/2d_contour_nova_vanilla/ \
--dir3 my_results/pingu_v47/3_year_contours/2d_contour_t2k_better_sys_res \
--dir4 my_results/pingu_v47/3_year_contours/2d_contour_t2k_vanilla/ \
-x theta23 -y deltam31 -t 3year \
-t1 "NOvA injected, optimistic" \
-t2 "NOvA injected, baseline" \
-t3 "T2K injected, optimistic" \
-t4 "T2K injected, baseline"
