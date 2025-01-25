
set datafile separator ','    
#set terminal wxt enhanced size 800,800
set title 'Hofstadter Butterfly'
set xlabel 'Alpha'
set ylabel 'Energy'
set grid
#set xrange [0:0.2]
plot 'dataHofstadterButterfly.dat' using 1:2 with points pt 7 ps 0.2 lc rgb 'red' title '1 band q = 50'
save '1band_qmax=50.pdf'
pause -1
