
set datafile separator ','
set title 'Hofstadter Butterfly'
set xlabel 'Alpha'
set ylabel 'Energy'
set grid
plot '3band_dataHofstadterButterfly_q_151.dat' u 1:2 with points pt 8 ps 0.2 lc rgb 'red' notitle
pause -1
        