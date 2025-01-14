

import sounddevice as sd
from scipy.io.wavfile import write

fs = 48000  # Sample rate
seconds = 3  # Duration of recording
sd.default.device=12
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=16)
sd.wait()  # Wait until recording is finished

write("C:/Users/arg/Documents/GitHub/EyeHear/Acoustics/AudioTests/test10.wav", fs, myrecording)  # Save as WAV file 
