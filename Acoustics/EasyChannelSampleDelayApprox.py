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
class ShiftCalc:
    def __init__(self,n_channels=8,spacing=np.array([[0,0],[0.028,0],[0.056,0],[0.084,0],[0.112,0],[0.14,0],[0.168,0],[0.196,0]]),sample_rate=48000):
        self.n_channels = n_channels
        self.spacing = spacing
        self.sample_rate = sample_rate
        self.delays = np.zeros(n_channels) #in microseconds
        self.gains = np.ones(n_channels) # multiplier
        self.sample_dur= 1/sample_rate *10**6 #Duration of a sample in microseconds
        self.delay_approx=DelayAproximator(self.spacing)

 
    #calculates number of samples to delay
    def calculate_channel_shift(self):
        channel_shifts=((self.delays/self.sample_dur))        
        return channel_shifts
    def update_delays(self,doa): #doa in degrees, assuming plane wave as it is a far-field source
        
        self.delays=np.array(self.delay_approx.get_delays(DelayAproximator.get_pos(doa,20000)))*10**6

    def update_gains(self,distance):
        for i in range(self.n_channels):
            self.gains[i]=1/distance**2

# Speed of sound si 343 m/s
sampledelay=np.array([[4,6],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[9,11]]) # Spacing (x,y) in meters 
sig_gen=ShiftCalc(8,sampledelay*343/48000)
sig_gen.update_delays(90) # 0 = +y direction   -90 = +x direction +90 = -x direction
print(np.round(sig_gen.calculate_channel_shift())) # Relative to the first microphone in the array



