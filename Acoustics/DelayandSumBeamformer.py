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
        self.update_delays(0,0)
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

    def update_delays(self,azimuth,elevation): #doa in degrees, assuming plane wave as it is a far-field source
        # self.doa=doa
        self.delays=np.array(self.delay_approx.get_flat_delays(azimuth,elevation))*10**6
        print(self.delays)
        shift=min(self.delays)
        self.delays+=-shift
        print("azi ele")
        print(str(azimuth)+" "+str(elevation))
        print("channel shift")
        print(self.calculate_channel_shift())


from SignalGen import SignalGen
from Preprocessor import Preprocessor
import soundfile as sf
# spacing=np.array([[-0.1,-0.1,0],[-0.1,0.0,0],[-0.1,0.1,0],[0,-0.1,0],[0,0,0],[0,0.1,0],[0.1,-0.1,0],[0.1,0,0],[0.1,0.1,0]])
# spacing=np.array([[-0.2,-0.2,0],[-0.2,-0.1,0],[-0.2,0.1,0],[-0.2,0.2,0],[-0.1,-0.2,0],[-0.1,-0.1,0],[-0.1,0.1,0],[-0.1,0.2,0],[0.1,-0.2,0],[0.1,-0.1,0],[0.1,0.1,0],[0.1,0.2,0],[0.2,-0.2,0],[0.2,-0.1,0],[0.2,0.1,0],[0.2,0.2,0]])
# spacing=np.array([[-0.18,0.12,0],[-0.06,0.12,0],[0.06,0.12,0],[0.18,0.12,0],[-0.18,0,0],[-0.06,0,0],[0.06,0,0],[0.18,0,0],[-0.18,-0.12,0],[-0.06,-0.12,0],[0.06,-0.12,0],[0.18,-0.12,0]])
# spacing=np.array([[-0.06,-0.12,0],[-0.18,-0.12,0],[-0.06,0,0],[-0.18,0,0],[-0.06,0.12,0],[-0.18,0.12,0],[0.18,-0.12,0],[0.06,-0.12,0],[0.18,0,0],[0.06,0,0],[0.18,0.12,0],[0.06,0.12,0]])
spacing=np.array([[-0.06,-0.24,0],[-0.18,-0.24,0],[-0.06,-0.12,0],[-0.18,-0.12,0],[-0.06,0,0],[-0.18,0,0],[-0.06,0.12,0],[-0.18,0.12,0],[0.18,-0.24,0],[0.06,-0.24,0],[0.18,-0.12,0],[0.06,-0.12,0],[0.18,0,0],[0.06,0,0],[0.18,0.12,0],[0.06,0.12,0]])

# sig=Sine(1500,0.5,48000)
azimuth=60
elevation=60
pe=Preprocessor(interpolate=3)
target_samplerate=48000
sig_gen=SignalGen(16,spacing)
speech,samplerate=sf.read(("C:/Users/arg/Documents/GitHub/EyeHear/Acoustics/AudioTests/test8.wav"))
# interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
# print(speech.shape)
# speech=np.reshape(speech,(-1,1))
# print(speech.shape)
# speech=interpolator.process(speech)
# print(speech.shape)
# sig_gen.update_delays(azimuth,elevation)
# print(speech.shape)
# angled_speech=sig_gen.delay_and_gain(speech)
# print(angled_speech.shape)
# speech1,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/652/130737/652-130737-0005.flac"))
# interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
# speech1=np.reshape(speech1,(-1,1))
# speech1=interpolator.process(speech1)
# sig_gen.update_delays(50,-80)
# # angled_speech=angled_speech[0:min(len(speech),len(speech1))]+sig_gen.delay_and_gain(speech1)[0:min(len(speech),len(speech1))]*1.2
# angled_speech=sig_gen.delay_and_gain(speech1)


# spacing=np.array([[-0.1,-0.1,0],[-0.1,0.0,0],[-0.1,0.1,0],[0,-0.1,0],[0,0,0],[0,0.1,0],[0.1,-0.1,0],[0.1,0,0],[0.1,0.1,0]])
beam=Beamformer(n_channels=16,coord=spacing)
segments=21
rms_data=np.zeros((segments,segments))
azi=-90
ele=-90
for i in range(segments):
    for j in range(segments):
        
        io=IOStream()
        io.arrToStream(speech,48000)
        print(i)
        beam.update_delays(azi+(180/segments)/2,ele+(180/segments)/2)
        while(not io.complete()):
            
            sample=io.getNextSample()
            sample[np.abs(sample) < (0.00063)] = 0
            outdata=beam.beamform(sample)
            rms_data[i][j]+=np.mean(outdata**2)
        ele+=180/segments
        print(rms_data)
    ele=-90
    azi+=180/segments
print(rms_data)
# rms_data=np.array([[37.09039681, 37.09039732, 37.09039732, 37.09039732], [26.2475326,  37.46160062, 64.05876732, 75.13930766], [37.09039713, 64.0587671,  99.82954105, 54.64336452], [45.31064137, 70.59155776, 54.64336426, 29.43501057]])

import numpy as np
import matplotlib.pyplot as plt


segments = rms_data.shape[0]
azi_angles = np.linspace(-90, 90, segments)  # Azimuth angles
ele_angles = np.linspace(-90, 90, segments)  # Elevation angles

# Create a meshgrid for azimuth and elevation
azi_grid, ele_grid = np.meshgrid(azi_angles, ele_angles)

# Normalize RMS data for visualization
rms_data_normalized = rms_data / np.max(rms_data)

# Plot the heatmap
plt.figure(figsize=(10, 7))
plt.pcolormesh(azi_grid, ele_grid, rms_data_normalized, shading='auto', cmap='viridis')
plt.colorbar(label="Normalized RMS Power")
plt.xlabel("Azimuth (°)")
plt.ylabel("Elevation (°)")
plt.title("RMS Power Distribution (Azimuth-Elevation Plane)")

# Highlight maximum RMS power point
max_idx = np.unravel_index(np.argmax(rms_data), rms_data.shape)
max_azi = azi_angles[max_idx[1]]
max_ele = ele_angles[max_idx[0]]
plt.scatter(max_azi, max_ele, color='red', label="Strongest RMS Power", zorder=5)
plt.legend()

plt.show()




