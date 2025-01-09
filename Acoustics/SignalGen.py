import numpy as np
from Signal import Sine,Sawtooth,Square,Signal,Chirp
import Signal

from scipy import signal
v=336
from DelayApproximation import DelayAproximator


# def angleToWav(angles,signals,n_mics,spacing,samplerate=48000):
#     for i in range(len(signals)):
#         signal=signals[i]
#         doa=angles[i]
        
        
class SignalGen:
    def __init__(self,n_channels=8,spacing=np.array([[0,0],[0.028,0],[0.056,0],[0.084,0],[0.112,0],[0.14,0],[0.168,0],[0.196,0]]),sample_rate=48000):
        self.n_channels = n_channels
        self.spacing = spacing
        self.sample_rate = sample_rate
        self.delays = np.zeros(n_channels) #in microseconds
        self.gains = np.ones(n_channels) # multiplier
        self.sample_dur= 1/sample_rate *10**6 #Duration of a sample in microseconds
        self.delay_approx=DelayAproximator(self.spacing)
        # print(spacing)

    def delay_and_gain(self, samples):
        #ok i really need to make this use fft so i can change the complexity form n^2 to nlogn or something
        shifts=self.calculate_channel_shift()

        # print("Delays")
        # print(shifts)
        intshifts=np.floor(shifts)
        # print(intshifts)
        max_sample_shift=int(max(shifts))
        # print(max_sample_shift)
        dims = samples.shape
        dims=(int(dims[0]+max_sample_shift),dims[1])
        delayed = np.zeros((dims[0],self.n_channels))
        for i in range(self.n_channels):
            intermult=1-(shifts[i]%1)
            shiftdiff=int(intshifts[i])
            for j in range(1,dims[0]-max_sample_shift):
                delayed[j+shiftdiff][i]=self.gains[i]*((samples[j][0]-samples[j-1][0])*(intermult)+samples[j-1][0])               
            
        
        return delayed
    #calculates number of samples to delay
    def calculate_channel_shift(self):
        channel_shifts=((self.delays/self.sample_dur))
        # print(channel_shifts)
        
        return channel_shifts
    def update_delays(self,azimuth,elevation): #doa in degrees, assuming plane wave as it is a far-field source
        
        self.delays=np.array(self.delay_approx.get_delays(DelayAproximator.get_pos(azimuth,elevation,20000)))*10**6
        # shift=min(self.delays)
        # self.delays+=-shift
        # print(self.delays)
    def update_gains(self,distance):
        for i in range(self.n_channels):
            self.gains[i]=1/distance**2


# approx=DelayAproximator([[0,0],[0.028,0],[0.056,0],[0.084,0],[0.112,0],[0.14,0],[0.168,0],[0.196,0]])
# pos=[1.414,1.414]
# delays=approx.get_delays(pos)
# print(delays)
# gen= SignalGen()
# gen.update_delays(45)
# print(gen.delays/10**6)
# gen=SignalGen()
# writer= AudioWriter()
# # sine=Chirp(start_freq=50,end_freq=150)
# sine=Sine(frequency=50)
# sine_data=sine.generate_wave(1)
# gen.update_delays(0)
# sine_data=gen.delay_and_gain(sine_data)
# writer.add_sample(sine_data,0)
# writer.write("./beamformingarray/AudioTests/1gen.wav",48000)
# stream=IOStream()
# stream.arrToStream(sine_data,48000)
# cc=CrossCorrelatior(spacing=0.03)
# # sample= stream.getNextSample()
# # for i in range(10):
# #     stream.getNextSample()
# sample= stream.getNextSample()
# # window1 = signal.windows.hamming(len(sample.T[0]))
# # print(len(window1))
# # cross_corr=signal.correlate(window1*sample.T[0],window1*sample.T[1])
# # lags=signal.correlation_lags(len(sample.T[0]),len(sample.T[1]))
# # lag=lags[np.argmax(cross_corr)]
# # print(lag)
# print(cc.get_doa(sample,True))
# # for i in range(50):
# print(cc.get_doa(stream.getNextSample(),True))
# print(cc.get_doa(stream.getNextSample(),True))
# print(cc.get_doa(stream.getNextSample(),True))
# for i in range(len(sample)):
#     print(str(i)+" "+ str(sample[i]))
# while(not stream.complete()):
#     print(cc.get_doa(stream.getNextSample(),True))
# splicer =SignalSplicer(48000)
# gen=SignalGen(spacing=0.03)
# writer= AudioWriter()
# sine=Sine(frequency=50)
# sine_data=sine.generate_wave(1)
# saw=Sawtooth(frequency=4,amplitude=0.4)
# saw_data=saw.generate_wave(0.6)
# gen.update_delays(0)
# sine_data=gen.delay_and_gain(sine_data)
# # gen.update_delays(135)
# # saw_data=gen.delay_and_gain(saw_data)
# # sum_data = Signal.sum_signals(saw_data,sine_data)
# # print(sum_data.shape)
# # sum_data=Signal.add_noise(sine_data,noise_level=0.2)
# # print(sum_data.shape)
# writer.add_sample(sine_data)
# writer.write("./beamformingarray/gentest17.wav",48000)
# sine.sum(Sawtooth(48000,0,period=6)) # Sum
# gen.update_delays(0)
# sig1=Signal(gen.delay_and_gain(sine.data),48000,0,t=2)
# gen.update_delays(90)
# saw=Sawtooth(48000,0,period=1,amp=0.5)
# sig1.sum(Signal(gen.delay_and_gain(saw.data),48000,0,t=2))
# writer.add_sample(sig1.data)
# writer.write("./beamformingarray/gentest14.wav",48000) 
# writer2=AudioWriter()
# writer2.add_sample(sine.data)
# writer2.write("./beamformingarray/gentest14o.wav",48000)
# saw=Sawtooth(48000,100000,period=1)
# splicer =SignalSplicer(48000)
# sine=Sawtooth(48000,0,period=2,length=1)
# splicer.add_signal(sine)
# sine=Sine(48000,0,period=6,length=1)
# splicer.add_signal(sine)
# sine=Square(48000,0,period=6,length=1)
# splicer.add_signal(sine)
# sine=Sawtooth(48000,0,period=1,length=1)
# splicer.add_signal(sine)
# splicer.write("./beamformingarray/splicetest1.wav")