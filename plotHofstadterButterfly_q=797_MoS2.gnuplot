
set terminal wxt size 700,900 
set datafile separator ','
set title 'Hofstadter Butterfly'
set xlabel 'Alpha'
set ylabel 'Energy'
set grid
#set xrange [0:0.12]
#set yrange [1.3:2.0]
plot '3band_dataHofstadterButterfly_q_797_MoS2_test.dat' u 1:2 with points pt 7 ps 0.3 lc rgb 'black' notitle 'HofstadterButterfly_MoS2'
pause -1
        