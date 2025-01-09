import numpy as np
from Signal import *
from scipy.io.wavfile import write,read
from SignalGen import SignalGen
from Preprocessor import Preprocessor
import soundfile as sf
from VAD import VAD
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
speech,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/2035/147961/2035-147961-0018.flac"))
chunk_size=int(samplerate/50);
end=chunk_size;
start=0
vad=VAD()
fund=[]
energy=[]
while end<=len(speech) :
    e=np.sum(np.square(speech[start:end]))
    if vad.is_speech(speech[start:end]) :
        # Number of samplepoints
        N =int( chunk_size)
        # sample spacing
        T = 1.0 / samplerate
        x = np.linspace(0.0, N*T, N)
        y = speech[start:end]
        yf = scipy.fftpack.fft(y,1024)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        magnitude = 2.0 / N * np.abs(yf[:N // 2])
        print(len(xf))
        print("Fadsfaksdhfuajksdhfjkasdh fjkasdjkfasdkjfasdjkfasdksd")
        peaks, _ = scipy.signal.find_peaks(magnitude[1:], height=e/800)
        peaks+=1
        if(len(peaks) == 0):
            fund.append(-1)
            continue
        first_peak_freq = xf[peaks[0]]
        fund.append(first_peak_freq)
        # Find the multiples of the first peak
        multiples = []
        for multiple in range(2, int(xf[-1] // first_peak_freq) + 1):
            multiple_freq = multiple * first_peak_freq
            closest_idx = np.argmin(np.abs(xf - multiple_freq))
            multiples.append(closest_idx) 
        fig, ax = plt.subplots()
        ax.plot(xf, magnitude)
        ax.plot(xf[peaks], magnitude[peaks], 'ro', label='Peaks')
        ax.plot(xf[multiples], magnitude[multiples], 'go', label='Multiples of First Peak')  # Highlight multiples with green x's
        for i in multiples:
            magnitude[i]=0
        ax.legend()
        plt.show()
    else:
        fund.append(0)
        pass
    energy.append(10*e)
    end+=chunk_size
    start+=chunk_size
    print(end/len(speech))
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# Function to plot the spectrogram
def plot_spectrogram(ax, audio_data, sample_rate):
    f, t, Sxx = spectrogram(audio_data, sample_rate)
    ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title('Spectrogram')

# Function to plot the amplitude over time
def plot_amplitude(ax, audio_data, sample_rate):
    time = np.arange(0, len(audio_data)) / sample_rate
    ax.plot(time, audio_data)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time [sec]')
    ax.set_title('Amplitude over Time')

# Function to plot additional data (user-supplied)
def plot_custom_data(ax, custom_data, title='Custom Data'):
    ax.plot(custom_data)
    ax.set_ylabel('Value')
    ax.set_xlabel('Index')
    ax.set_title(title)

# Load an example audio file (replace 'example.wav' with your audio file)
audio_data = speech

# Custom data (replace with your own data)
custom_data = fund

# Create a figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot each graph on its respective subplot
plot_spectrogram(axs[1], audio_data, samplerate)
plot_amplitude(axs[0], audio_data, samplerate)
plot_custom_data(axs[2], custom_data, title='Fundamentals')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
