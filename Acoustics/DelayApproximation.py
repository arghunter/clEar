# Make something where oyu pass in source coords and the coord of the mics and then i calculates the time diff from each mic from the first mic
import numpy as np
v=343
class DelayAproximator:

    def __init__(self,coords) :
        self.coords=coords

    def get_delays(self,pos):
        distances = []
        for mic_pos in self.coords:
            dx = mic_pos[0] - pos[0]
            dy = mic_pos[1] - pos[1]
            
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)
        reference_distance = distances[0]
        tdoa_values = []
        for distance in distances:
            time_diff = (distance - reference_distance) / v
            tdoa_values.append(time_diff)
        return tdoa_values
    def get_pos(angle,dist):
        angle=np.radians(angle)
        pos=[dist*np.cos(np.pi/2-angle),dist*np.sin(np.pi/2-angle)]
        # print(pos)
        return pos
    # def get_pos(azimuth,elevation,dist):
        # azimuth=np.radians(azimuth+90)
        # elevation=np.radians(elevation)
        # pos=[dist*np.cos(azimuth)*np.cos(elevation),dist*np.cos(azimuth)*np.sin(elevation),dist*np.sin(azimuth)]
        # print(pos)
        # return pos
    def get_flat_delays(self,azimuth,elevation):
        delays=np.zeros((len(self.coords)))
        delayx=np.zeros((len(self.coords)))
        delayy=np.zeros((len(self.coords)))
        iter=0
        for mic_pos in self.coords:
            delayx[iter]=(((np.cos(np.radians(azimuth)+np.pi/2)*mic_pos[0])/v))
            delayy[iter]=(((np.cos(np.radians(elevation)+np.pi/2)*mic_pos[1])/v))
            iter+=1
        delayx+=min(delayx)
        delayy+=min(delayy)
        iter=0
        delays=np.sqrt(delayx**2+delayy**2)
        
            
        return delays
    


