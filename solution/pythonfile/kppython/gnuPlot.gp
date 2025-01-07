
            
set multiplot layout 1, 2
set grid


set style line 1 linetype 1 dashtype 2 lc rgb "blue" lw 3
set style line 2 linetype 1 lc rgb "red" lw 3
set style line 3 linetype 1 lc rgb "black" lw 3
#set xrange [-0.1:0.1]

plot "eigenvaluesValence.csv" using 1:2 with lines linestyle 1 title "k^1",      "eigenvaluesValence.csv" using 1:3 with lines linestyle 2 title "k^2",      "eigenvaluesValence.csv" using 1:4 with lines linestyle 3 title "k^3"

plot "eigenvaluesConduction.csv" using 1:2 with lines linestyle 1 title "k^1",      "eigenvaluesConduction.csv" using 1:3 with lines linestyle 2 title "k^2",      "eigenvaluesConduction.csv" using 1:4 with lines linestyle 3 title "k^3", 

pause -1
