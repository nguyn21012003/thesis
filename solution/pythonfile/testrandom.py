import pybinding as pb
import tmdybinding as td
import matplotlib.pyplot as plt
from tmdybinding.parameters import liu6


# Construct the unit cell
lat = td.TmdNN12MeoXeo(params=liu6["MoS2"], soc=True).lattice()

# Make a periodic system
model = pb.Model(lat, pb.translational_symmetry())

# Get the corners from the Brillouin zone
bz = lat.brillouin_zone()

# Calculate along the high symmetry path
bands = pb.solver.lapack(model).calc_bands(bz[3] * 0, bz[3], (bz[3] + bz[4]) / 2, bz[3] * 0)

# Visualize the results
# bands.plot(point_labels=[r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])

lat.plot()
plt.show()
