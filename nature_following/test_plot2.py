import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Imported pyplot")
fig, ax = plt.subplots()
print("Created subplots")
fig.savefig("test2.png")
print("Saved PNG")
fig.savefig("test2.svg")
print("Saved SVG")
