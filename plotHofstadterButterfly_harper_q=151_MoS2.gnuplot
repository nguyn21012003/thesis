
set terminal wxt size 700,900 
set datafile separator ','
set title 'HofstadterButterfly MoS2'
set xlabel 'Alpha'
set ylabel 'Energy'
set grid
#set xrange [0:0.12]
#set yrange [1.3:2.0]
plot '1band_dataHofstadterButterfly_harper_q_151_h0.dat' u 1:2 with points pt 7 ps 0.3 lc rgb 'black' title 'HofstadterButterfly MoS2'
pause -1
        