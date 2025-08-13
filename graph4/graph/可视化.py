import matplotlib.pyplot as plt
import json

with open('density.json', 'r') as f:
    matrix_people = json.load(f)
plt.figure(figsize = (30,30))
plt.imshow(matrix_people,
           cmap = 'viridis',
           interpolation = 'bilinear',
           aspect = "auto",
           )
plt.colorbar()
plt.show()

