import numpy as np
import matplotlib.pyplot as plt
import wave
import sys

# Function to read a WAV file and extract channels
def read_wav_file(filename):
    try:
        wav = wave.open(filename, 'r')
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        framerate = wav.getframerate()
        
        # Read frames and convert to numpy array
        frames = wav.readframes(n_frames)
        wav.close()

        # Convert byte data to numpy array
        data = np.frombuffer(frames, dtype=np.int16).copy()  # Make a writable copy
        data[np.abs(data) < (0.00063*32768)] = 0  # Apply thresholding
        data = data.reshape(-1, n_channels)
        return data, framerate
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        sys.exit(1)

# Function to compute and display FFT for each channel
def plot_fft(data, framerate):
    n_channels = data.shape[1]
    n_samples = data.shape[0]

    for channel in range(n_channels):
        # Select channel data
        channel_data = data[:, channel]

        # Perform FFT
        fft_data = np.fft.fft(channel_data)
        freq = np.fft.fftfreq(n_samples, d=1/framerate)

        # Only take the positive half of the spectrum
        pos_mask = freq >= 0
        freq = freq[pos_mask]
        fft_data = np.abs(fft_data[pos_mask])

        # Plot the frequency spectrum
        plt.figure()
        plt.plot(freq, fft_data)
        plt.title(f"Channel {channel + 1} Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()

    plt.show()

if __name__ == "__main__":
    # Input WAV file
    filename = "C:/Users/arg/Documents/GitHub/EyeHear/Acoustics/AudioTests/test6noise.wav"

    # Read WAV file
    data, framerate = read_wav_file(filename)

    # Plot FFT for each channel
    plot_fft(data, framerate)