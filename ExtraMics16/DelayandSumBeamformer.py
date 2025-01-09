import numpy as np

from Preprocessor import Preprocessor
from time import time
import sounddevice as sd
from scipy import signal
from DelayApproximation import DelayAproximator
from IOStream import IOStream
from VAD import VAD
v=340.3 # speed of sound at sea level m/s
    
class Beamformer:
    def __init__(self,n_channels=8,coord=np.array([[-0.08,0.042],[-0.08,0.014],[-0.08,-0.028],[-0.08,-0.042],[0.08,0.042],[0.08,0.014],[0.08,-0.028],[0.08,-0.042]]),sample_rate=48000):
        self.n_channels = n_channels
        self.coord = coord
        self.sample_rate = sample_rate
        self.delays = np.zeros(n_channels) #in microseconds
        self.gains = np.ones(n_channels) # multiplier
        self.sample_dur= 1/sample_rate *10**6 #Duration of a sample in microseconds
        self.delay_approx=DelayAproximator(self.coord)
        self.doa=0
        self.update_delays(self.doa)
        self.locked=False
    def beamform(self,samples):
        
        sample_save=samples
        samples,max_sample_shift=self.delay_and_gain(samples)
       
        samples=self.sum_channels(samples)
        if hasattr(self,'last_overlap'):
            for i in range(self.last_overlap.shape[0]):
                samples[i]+=self.last_overlap[i]
        
        self.last_overlap=samples[samples.shape[0]-max_sample_shift:samples.shape[0]]
 
        

        return samples[0:samples.shape[0]-max_sample_shift]

    def sum_channels(self,samples):
        summed=np.zeros(samples.shape[0])
        for j in range(samples.shape[0]):
            summed[j] = samples[j].sum()
        return summed
    def delay_and_gain(self, samples):
        #backwards interpolations solves every prblem
        shifts=self.calculate_channel_shift()

        intshifts=np.floor(shifts)
        max_sample_shift=int(max(intshifts))
        dims = samples.shape
        dims=(int(dims[0]+max_sample_shift),dims[1])
        delayed = np.zeros(dims)
        if hasattr(self,'last_samples'):
            
            for i in range(self.n_channels):
                intermult=1-(shifts[i]%1)
                shiftdiff=max_sample_shift-int(intshifts[i])
                delayed[0+shiftdiff][i]=self.gains[i]*((samples[0][i]-self.last_samples[len(self.last_samples)-1][i])*(intermult)+self.last_samples[len(self.last_samples)-1][i])               
        else:
            for i in range(self.n_channels):
                intermult=1-(shifts[i]%1)
                shiftdiff=max_sample_shift-int(intshifts[i])
                delayed[0+shiftdiff][i]=(self.gains[i]*(samples[0][i]-0)*(intermult))               
        
        for i in range(self.n_channels):
            intermult=1-(shifts[i]%1)
            shiftdiff=max_sample_shift-int(intshifts[i])
            for j in range(1,dims[0]-max_sample_shift):
                delayed[j+shiftdiff][i]=self.gains[i]*((samples[j][i]-samples[j-1][i])*(intermult)+samples[j-1][i])               
            
        
        self.last_samples=samples
       
        return delayed,max_sample_shift
    #calculates number of samples to delay
    def calculate_channel_shift(self):
        channel_shifts=(self.delays/self.sample_dur)
        return channel_shifts

    def update_delays(self,doa): #doa in degrees, assuming plane wave as it is a far-field source
        self.doa=doa
        self.delays=np.array(self.delay_approx.get_delays(DelayAproximator.get_pos(doa,3)))*10**6
        shift=min(self.delays)
        self.delays+=-shift
        # print(self.calculate_channel_shift())

from SignalGen import SignalGen
from Preprocessor import Preprocessor
import soundfile as sf
spacing=np.array([[-0.08,0.042],[-0.08,0.014],[-0.08,-0.028],[-0.08,-0.042],[0.08,0.042],[0.08,0.014],[0.08,-0.028],[0.08,-0.042]])
# sig=Sine(1500,0.5,48000)
angle=114
pe=Preprocessor(interpolate=3)
target_samplerate=48000
sig_gen=SignalGen(8,spacing)
speech,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/2035/147961/2035-147961-0018.flac"))
interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
print(speech.shape)
speech=np.reshape(speech,(-1,1))
print(speech.shape)
speech=interpolator.process(speech)
print(speech.shape)
sig_gen.update_delays(angle)
print(speech.shape)
angled_speech=sig_gen.delay_and_gain(speech)

beam=Beamformer(coord=spacing)
segments=360
rms_data=np.zeros(segments)
doa=0
for i in range(segments):
    
    io=IOStream()
    io.arrToStream(angled_speech,48000)
    print(i)
    beam.update_delays(doa)
    while(not io.complete()):
        
        sample=io.getNextSample()
        outdata=beam.beamform(sample)
        rms_data[i]+=np.mean(outdata**2)
    doa+=360/segments
    print(rms_data)
print(rms_data)

import numpy as np
import matplotlib.pyplot as plt

# Example `rms_data` values and setup
segments = len(rms_data)
theta = np.linspace(0, 2 * np.pi, segments + 1)  # Add one to close the circle
rms_data_normalized = rms_data / np.max(rms_data)  # Normalize the RMS data
rms_data_normalized = np.append(rms_data_normalized, rms_data_normalized[0])  # Close the circle

# Extend theta to form the bins
theta_grid, radius_grid = np.meshgrid(theta, [0, 1])

# Extend the RMS data to fit the mesh
rms_data_grid = np.tile(rms_data_normalized, (2, 1))

# Create a filled polar heatmap
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
heatmap = ax.pcolormesh(theta_grid, radius_grid, rms_data_grid, cmap='jet', shading='auto')

# Add a marker for the sound source
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


