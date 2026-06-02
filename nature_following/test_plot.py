import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Imported pyplot")
fig, ax = plt.subplots()
print("Created subplots")
fig.savefig("test.pdf")
print("Saved figure")
