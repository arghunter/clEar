import numpy as np
import matplotlib.pyplot as plt

# Example RMS data (replace with your actual data)
rms_data = [0.45686734, 0.31404594, 0.37565332, 0.33370884, 0.075857, 0.18200582,
 0.41648925, 0.5312394,  0.52055039, 0.53118349, 0.43077909, 0.18486702,
 0.07654179, 0.3377602,  0.37555849, 0.32219962]
rms_data = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# Number of segments and distances (you can adjust this)
segments = len(rms_data)
distances = 100  # Number of distance levels (bins) per segment

# Generate theta values for the polar plot
theta = np.linspace(0, 2 * np.pi, segments + 1)  # Add one to close the circle
rms_data_normalized = rms_data / np.max(rms_data)  # Normalize the RMS data
rms_data_normalized = np.append(rms_data_normalized, rms_data_normalized[0])  # Close the circle

# Extend theta to form the bins
theta_grid, radius_grid = np.meshgrid(theta, np.linspace(0, 1, distances))

# Initialize a matrix for RMS data corresponding to each distance bin
rms_data_grid = np.tile(rms_data_normalized, (distances, 1))
for i in range(segments):
    for j in range(distances):
        
        rms_data_grid[j][i]=0
# for i in range(segments):
#     for j in range(segments):
        
#         rms_data_grid[int(rms_data_normalized[i]*distances)-1][j]=rms_data_normalized[i]
for i in range(segments+1):
        for j in range(int((1-np.sqrt(rms_data_normalized[i]))*(distances-1)),distances):
            rms_data_grid[j][i]=rms_data_normalized[i]
        for j in range(int((1-np.sqrt(rms_data_normalized[i]))*(distances-1))):
            rms_data_grid[j][i]=rms_data_normalized[i]/(((np.sqrt(rms_data_normalized[i])))/((float(distances)-j)/distances))**2
            # rms_data_grid[j][i]=rms_data_normalized[i]*((float(distances)-j)/distances)**2


# Create a filled polar heatmap
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
heatmap = ax.pcolormesh(theta_grid, radius_grid, rms_data_grid, cmap='jet', shading='auto')

# Add a marker for the sound source
angle = 60  # Example angle
source_theta = np.deg2rad(angle)
ax.plot(source_theta, 1, 'ro', label=f'Sound Source ({angle}°)', markersize=10)

# Add a colorbar
cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical')
cbar.set_label('Normalized RMS Power')

# Add labels, legend, and title
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
ax.set_title('Circular Heatmap of RMS Power')
ax.grid(False)
ax.set_yticklabels([])

plt.show()
