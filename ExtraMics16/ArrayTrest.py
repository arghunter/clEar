import numpy as np
from Signal import *
from scipy.io.wavfile import write,read
from SignalGen import SignalGen
from Preprocessor import Preprocessor
import soundfile as sf
sampledelay=np.array([[0,0],[0,4],[0,10],[0,12],[0,14],[0,18],[3,0],[9,0],[15,0],[21,0],[24,0],[24,4],[24,10],[24,12],[24,14],[24,18]]) 
# sig=Sine(1500,0.5,48000)
pe=Preprocessor(interpolate=3)
target_samplerate=48000
sig_gen=SignalGen(16,sampledelay*343/48000)
speech,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/2035/147961/2035-147961-0018.flac"))
interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
speech=np.reshape(speech,(-1,1))
speech=interpolator.process(speech)
sig_gen.update_delays(0)
# angled_speech=sig_gen.delay_and_gain(speech)*0.00
# speech1,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/652/130737/652-130737-0005.flac"))
# interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
# speech1=np.reshape(speech1,(-1,1))
# speech1=interpolator.process(speech1)
# sig_gen.update_delays(90)
# angled_speech=angled_speech[0:min(len(speech),len(speech1))]+sig_gen.delay_and_gain(speech1)[0:min(len(speech),len(speech1))]
# speech2,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/6319/64726/6319-64726-0016.flac"))
# interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
# speech2=np.reshape(speech2,(-1,1))
# speech2=interpolator.process(speech2)
# sig_gen.update_delays(0)
# angled_speech=angled_speech[0:min(len(speech),min(len(speech1),len(speech2)))]+sig_gen.delay_and_gain(speech2)[0:min(len(speech),min(len(speech1),len(speech2)))]
# angled_speech+=0.05*np.random.randn(*angled_speech.shape)
# samplesdelayed=np.zeros((16,48000*5))
# for i in range(16):
# sig_gen=SignalGen(16,sampledelay*343/48000)
# sig_gen.update_delays(0)
# angled_speech=sig_gen.delay_and_gain(sig.generate_wave(1))
# sig=Sine(1600,0.5,48000)
# sig_gen.update_delays(180)
# angled_speech+=sig_gen.delay_and_gain(sig.generate_wave(1))[0:48000]
#     samplesdelayed[i]=np.roll(wave,int(sampledelay[i][0]))
# noise = np.random.normal(0,1,angled_speech.shape)*0.1
# angled_speech+=noise
samplesdelayed=angled_speech.T

# print(samplesdelayed.dtype)
write("ExtraMics16/AudioTests/d1b.wav", 48000,samplesdelayed.T)
samplesmerged=np.zeros((16,198480))
summed=np.zeros((1,198480))
for i in range(16):
    summed+=samplesdelayed[i]
    samplesmerged[0]+=np.roll(samplesdelayed[i],-int(sampledelay[i][0]))# front
    samplesmerged[1]+=np.roll(samplesdelayed[i],int(sampledelay[i][0]))#back
    samplesmerged[2]+=np.roll(samplesdelayed[i],-int(sampledelay[i][1]))# Left (maybe)
    samplesmerged[3]+=np.roll(samplesdelayed[i],int(sampledelay[i][1]))#Right(maybe)
samplesmerged/=4 
samplesmerged[4]+=np.roll(samplesdelayed[1],int(-5))+ samplesdelayed[6]+   np.roll(samplesdelayed[3],int(-15)) +samplesdelayed[7]
samplesmerged[5]+=np.roll(samplesdelayed[1],int(5))+ samplesdelayed[6]+   np.roll(samplesdelayed[3],int(15)) +samplesdelayed[7]
samplesmerged[6]+=np.roll(samplesdelayed[11],int(-5))+ samplesdelayed[9]+   np.roll(samplesdelayed[13],int(-15)) +samplesdelayed[8]
samplesmerged[7]+=np.roll(samplesdelayed[11],int(5))+ samplesdelayed[9]+   np.roll(samplesdelayed[13],int(15)) +samplesdelayed[8]
samplesmerged[8]+=np.roll(samplesdelayed[5],int(-30))+ samplesdelayed[10]+   np.roll(samplesdelayed[4],int(-25)) +samplesdelayed[9]
samplesmerged[9]+=np.roll(samplesdelayed[5],int(30))+ samplesdelayed[10]+   np.roll(samplesdelayed[4],int(25)) +samplesdelayed[9]
samplesmerged[10]+=np.roll(samplesdelayed[15],int(-30))+ samplesdelayed[0]+   np.roll(samplesdelayed[14],int(-25)) +samplesdelayed[6]
samplesmerged[11]+=np.roll(samplesdelayed[15],int(30))+ samplesdelayed[0]+   np.roll(samplesdelayed[14],int(25)) +samplesdelayed[6]
samplesmerged[12]+=np.roll(samplesdelayed[2],int(-26))+ samplesdelayed[10]+   np.roll(samplesdelayed[4],int(-26)) +samplesdelayed[11]
samplesmerged[13]+=np.roll(samplesdelayed[2],int(26))+ samplesdelayed[10]+   np.roll(samplesdelayed[4],int(26)) +samplesdelayed[11]
samplesmerged[14]+=np.roll(samplesdelayed[14],int(-26))+ samplesdelayed[2]+   np.roll(samplesdelayed[12],int(-26)) +samplesdelayed[0]
samplesmerged[15]+=np.roll(samplesdelayed[14],int(26))+ samplesdelayed[2]+   np.roll(samplesdelayed[12],int(26)) +samplesdelayed[0]

# Now the speical ones  
# 53 degres: 1 6 3 7  dist 15 5
# other 53 : 9 11 8 13 dist 15 5
# mid degrere 5 10   4 9 dist 25.2 30
# other mid : 15 0 14 6 dist 25.2 30
# edge degrre: 2 10 4 11 dist 26 26
# other edge: 14 2 12 0  dist 26 26
samplesmerged/=4
summed/=16
write("ExtraMics16/AudioTests/d1p.wav", 48000,samplesmerged.T)
# samplesmerged/=16
# from sklearn.decomposition import FastICA
# for j in range(15):
#     ica = FastICA(n_components=1, random_state=0)
#     separated_sources = ica.fit_transform(samplesmerged[i:i+2].T).T
#     # separated_sources/=max(separated_sources)
#     for i, source in enumerate(separated_sources):
#         separated_sources[i]/=max(separated_sources[i])
#         sf.write(f'ExtraMics16/AudioTests/separated_source_{i+1}_{j+1}.wav', source, 48000)  # Assuming a sampling rate of 44100 Hz
# for i, source in enumerate(separated_sources):
#     plt.figure()
#     plt.plot(source)
#     plt.title(f'Separated Source {i+1}')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Number of samplepoints
N =int( 48000)
# sample spacing
T = 1.0 / 48000.0
x = np.linspace(0.0, N*T, N)
y = summed[0]
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots()
ax.plot(xf[0:4000], (2.0/N * np.abs(yf[:N//2]))[0:4000])
plt.show()
N =int( 48000)
# sample spacing
T = 1.0 / 48000.0
x = np.linspace(0.0, N*T, N)
y = samplesmerged[ 3]
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots()
ax.plot(xf[0:4000], (2.0/N * np.abs(yf[:N//2]))[0:4000])
plt.show()


N =int( 48000)
# sample spacing
T = 1.0 / 48000.0
x = np.linspace(0.0, N*T, N)
y = samplesmerged[ 2]
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots()
ax.plot(xf[0:4000], (2.0/N * np.abs(yf[:N//2]))[0:4000])
plt.show()
# Next steps
# Next goal: Waht i need is some way to take in the 16 directions and split them voices in real time
# 