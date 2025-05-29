import subprocess


def PlotMatrixGNU(fileMatrix, fileName):
    with open(fileName, "w") as GNUPLOT:
        GNUPLOT.write(
            f"""
set pm3d map
set size ratio -1
set palette defined (0 "white", 1 "sea-green", 2 "blue", 3 "yellow", 4 "orange" , 5 "red")
set xlabel "Column Index"
set ylabel "Row Index"
set yrange [*:*] reverse

set tics out

set xtics offset 0,1

set autoscale xfix
set autoscale yfix
set linestyle 81 lt 1 lw 9.0 lc rgb "black"

set xtics 20
set ytics 20


set grid


#set tics scale 0,0.001
#set mxtics 2
#set mytics 2
#set grid front mxtics mytics lw 1.5 lt -1 lc rgb 'white'

set label "(b)" at 2,55,10 font "Arial,16" front


unset key
splot "{fileMatrix}" using 1:2:3 with image
pause -1
        """
        )
    subprocess.run(["gnuplot", fileName])
    return None
