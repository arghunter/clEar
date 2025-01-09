

import numpy as np



class VAD:
    
 
        
    def is_speech(self,samples):
        # is_noise = 0
        
        energy = np.sum(np.square(samples))
        print("Energy: ", energy)
        if energy < 0.5:
            return False
        return True
# vad = webrtcvad.Vad()
# vad.set_mode(3)

# io = IOStream()
# writer= AudioWriter()
# io.wavToStream("./beamformingarray/output5_100v2res1.wav")
# arr=np.zeros(480)

# # print(edited)
# ag=np.ones(480)
# ab=np.zeros(480)
# for i in range(1000):
    
#     edited=np.int16(32767*io.getNextSample()).tobytes()
#     print(len(edited))
#     # print(edited)
#     # if vad.is_speech(edited, 48000):
#     #     writer.add_sample(ag)
#     # else:
#     #     writer.add_sample(ab)
#     # print(np.reshape(io.getNextSample(),(-1)).shape)
#     print ('Contains speech: %s' % (vad.is_speech(edited, 48000)))
# writer.write("./beamformingarray/output5_000v2res3.wav",1*48000)   
# sample_rate = 48000
# frame_duration = 10  # ms
# frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
# print(frame==edited)
# print ('Contains speech: %s' % (vad.is_speech(edited, sample_rate)))
# import webrtcvad
# import wave

# # Function to read audio data from a file
# def read_wave(path):
#     with wave.open(path, 'rb') as wf:
#         rate = wf.getframerate()
#         frames = wf.readframes(wf.getnframes())
#         return frames, rate

# # Function to detect voice activity
# def detect_voice_activity(audio_path):
#     audio, sample_rate = read_wave(audio_path)

#     vad = webrtcvad.Vad()
#     vad.set_mode(3)  # Set aggressiveness mode (0 to 3)

#     frame_duration = 30  # Duration of each frame in milliseconds
#     frame_length = int(sample_rate * (frame_duration / 1000.0))
    
#     # Process audio in frames
#     frames = []
#     for i in range(0, len(audio), frame_length):
#         frame = audio[i:i + frame_length]
#         if len(frame) < frame_length:
#             frame += b'\x00' * (frame_length - len(frame))  # Zero-padding for the last frame
#         is_speech = vad.is_speech(frame, sample_rate)
#         frames.append((i, is_speech))

#     return frames

# # Example usage
# audio_file_path = './beamformingarray/output5_000v2res1.wav'
# voice_activity = detect_voice_activity(audio_file_path)

# # Output the timestamps of voice activity
# for i, (start, is_speech) in enumerate(voice_activity):
#     print(f"Frame {i}: {'Speech' if is_speech else 'Silence'} at {start} ms")
