import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, frequency=1, amplitude=1, sample_rate=48000):
        self.frequency = frequency
        self.amplitude = amplitude
        self.sample_rate = sample_rate
    
    def generate_wave(self, duration):
        pass
    
    def plot_wave(self, duration):
        wave = self.generate_wave(duration)
        time = np.arange(0, duration, 1/self.sample_rate)
        plt.figure(figsize=(8, 4))
        plt.title(f"{self.__class__.__name__} Wave")
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.plot(time, wave)
        plt.show()
        return wave.reshape(-1, 1)
    


class Sine(Signal):
    def generate_wave(self, duration):
        num_samples = int(self.sample_rate * duration)
        time = np.linspace(0, duration, num_samples)
        sine_wave = self.amplitude * np.sin(2 * np.pi * self.frequency * time)
        return sine_wave.reshape(-1, 1)

class Square(Signal):
    def generate_wave(self, duration):
        num_samples = int(self.sample_rate * duration)
        time = np.linspace(0, duration, num_samples)
        square_wave = self.amplitude * np.sign(np.sin(2 * np.pi * self.frequency * time))
        return square_wave.reshape(-1, 1)

class Sawtooth(Signal):
    def generate_wave(self, duration):
        num_samples = int(self.sample_rate * duration)
        time = np.linspace(0, duration, num_samples)
        sawtooth_wave = self.amplitude * (2 * (time * self.frequency - np.floor(time * self.frequency + 0.5)))
        return sawtooth_wave.reshape(-1, 1)

class Chirp(Signal):
    def __init__(self, start_freq=20, end_freq=100, amplitude=1, sample_rate=48000):
        super().__init__(frequency=start_freq, amplitude=amplitude, sample_rate=sample_rate)
        self.start_freq = start_freq
        self.end_freq = end_freq
    
    def generate_wave(self, duration):
        num_samples = int(self.sample_rate * duration)
        time = np.linspace(0, duration, num_samples)
        chirp_wave = self.amplitude * np.sin(2 * np.pi * np.linspace(self.start_freq, self.end_freq, num_samples) * time)
        return chirp_wave.reshape(-1, 1)
def sum_signals(signal1_wave, signal2_wave):
   
    if len(signal1_wave) < len(signal2_wave):
            signal1_wave = np.pad(signal1_wave, ((0, len(signal2_wave) - len(signal1_wave)), (0, 0)), 'constant')
    elif len(signal1_wave) > len(signal2_wave):
            signal2_wave = np.pad(signal2_wave, ((0, len(signal1_wave) - len(signal2_wave)), (0, 0)), 'constant')
        
    summed_wave = signal1_wave + signal2_wave
       
        
    return summed_wave
def add_noise(signal_wave, noise_level=0.1):
        
   
    noise_shape = signal_wave.shape if len(signal_wave.shape) > 1 else (len(signal_wave), 1)
    
    noise = np.random.normal(0, noise_level, noise_shape)
    noisy_signal = signal_wave + noise

    
    return noisy_signal