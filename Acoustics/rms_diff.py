import numpy as np
import wave
import sys

# Function to read a WAV file and extract data and framerate
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

# Function to compute RMS power of a WAV file
def compute_rms(data):
    return np.sqrt(np.mean(data**2))
def save_wav_file(filename, data, framerate):
    """
    Save audio data to a WAV file.
    
    Parameters:
        filename (str): The name of the output WAV file.
        data (numpy.ndarray): The audio data as a NumPy array.
                             It should have shape (n_frames, n_channels).
        framerate (int): The sample rate (e.g., 44100 Hz).
    """
    try:
        # Ensure data is in the correct format
        if not isinstance(data, np.ndarray):
            raise ValueError("Audio data must be a NumPy array.")
        
        # Convert float data to int16 if necessary
        if data.dtype != np.int16:
            data = (data * 32768).astype(np.int16)

        # Flatten the data if it's multi-channel
        n_channels = data.shape[1] if data.ndim > 1 else 1
        data = data.flatten()

        # Open a WAV file for writing
        with wave.open(filename, 'w') as wav:
            wav.setnchannels(n_channels)       # Set number of channels
            wav.setsampwidth(2)               # Sample width in bytes (int16 = 2 bytes)
            wav.setframerate(framerate)       # Set sample rate
            wav.writeframes(data.tobytes())   # Write the audio data as bytes

        print(f"File '{filename}' saved successfully.")
    except Exception as e:
        print(f"Error saving WAV file: {e}")
if __name__ == "__main__":
    # Input WAV files
    filename1 = "C:/Users/arg/Documents/GitHub/EyeHear/Acoustics/AudioTests/test6speech.wav"
    filename2 = "C:/Users/arg/Documents/GitHub/EyeHear/Acoustics/AudioTests/test6noise.wav"

    # Read WAV files
    data1, framerate1 = read_wav_file(filename1)
    data2, framerate2 = read_wav_file(filename2)

    # Ensure both files have the same framerate and number of channels
    if framerate1 != framerate2:
        print("Error: WAV files have different sample rates.")
        sys.exit(1)
    if data1.shape[1] != data2.shape[1]:
        print("Error: WAV files have a different number of channels.")
        sys.exit(1)
    
    
    # data2[np.abs(data2) < 0.00003] = 0
    # Compute RMS power for each file
    rms1 = compute_rms(data1)
    rms2 = compute_rms(data2)

    # Subtract RMS power values
    rms_difference = rms1 - rms2

    # Output the result
    print(f"RMS Power of {filename1}: {rms1:.2f}")
    print(f"RMS Power of {filename2}: {rms2:.2f}")
    print(len(data1))
    print(f"Difference in RMS Power: {rms_difference:.2f}")
    save_wav_file('./beamformingarray/AudioTests/test8.wav', data1, 48000)