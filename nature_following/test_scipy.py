import sys
print("Starting...")

print("Importing scipy.signal...")
import scipy.signal

print("Importing scipy.ndimage...")
import scipy.ndimage

print("Importing scipy.spatial.distance...")
from scipy.spatial.distance import cdist, jensenshannon

print("Importing scipy.spatial.ConvexHull...")
from scipy.spatial import ConvexHull

print("Importing scipy.stats.spearmanr...")
from scipy.stats import spearmanr

print("All imports successful.")

import numpy as np
a = np.random.rand(100)
print("Testing uniform_filter1d...")
scipy.ndimage.uniform_filter1d(a, size=3)
print("Success.")

print("Testing ConvexHull...")
pts = np.random.rand(10, 2)
hull = ConvexHull(pts)
print("Success.")

print("Testing spearmanr...")
spearmanr(a, a)
print("Success.")

print("Testing jensenshannon...")
jensenshannon([0.5, 0.5], [0.5, 0.5])
print("Success.")

print("Done.")
