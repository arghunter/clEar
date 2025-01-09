import torch
import torchaudio
from torchaudio.models import ConvTasNet
import soundfile as sf
import numpy as np
from Signal import *
from scipy.io.wavfile import write,read
from SignalGen import SignalGen
from Preprocessor import Preprocessor
# Initialize the ConvTasNet model with default parameters
model = ConvTasNet(
    num_sources=2,  # Number of sources to separate
    enc_kernel_size=16,  # Encoder kernel size
    enc_num_feats=512,  # Number of encoder output channels
    msk_kernel_size=3,  # Masking network kernel size
    msk_num_feats=128,  # Number of intermediate channels in masking network
    msk_num_hidden_feats=512,  # Number of hidden channels in masking network
    msk_num_layers=8,  # Number of layers in masking network
    msk_num_stacks=3  # Number of stacks of masking network
)
sampledelay=np.array([[0,0],[0,4],[0,10],[0,12],[0,14],[0,18],[3,0],[9,0],[15,0],[21,0],[24,0],[24,4],[24,10],[24,12],[24,14],[24,18]]) 
pe=Preprocessor(interpolate=3)
target_samplerate=48000
sig_gen=SignalGen(16,sampledelay*343/48000)
speech,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/2035/147961/2035-147961-0018.flac"))
interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
speech=np.reshape(speech,(-1,1))
speech=interpolator.process(speech)
sig_gen.update_delays(0)
angled_speech=sig_gen.delay_and_gain(speech)
speech1,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/652/130737/652-130737-0005.flac"))
interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
speech1=np.reshape(speech1,(-1,1))
speech1=interpolator.process(speech1)
sig_gen.update_delays(90)
angled_speech=angled_speech[0:min(len(speech),len(speech1))]+sig_gen.delay_and_gain(speech1)[0:min(len(speech),len(speech1))]
# speech2,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/6319/64726/6319-64726-0016.flac"))
# interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
# speech2=np.reshape(speech2,(-1,1))
# speech2=interpolator.process(speech2)
# sig_gen.update_delays(0)
# angled_speech=angled_speech[0:min(len(speech),min(len(speech1),len(speech2)))]+sig_gen.delay_and_gain(speech2)[0:min(len(speech),min(len(speech1),len(speech2)))]
# Create a random input tensor (batch size, channels, samples)
# speech,samplerate=sf.read(("C:/Users/arg/Documents/Datasets/dev-clean.tar/dev-clean/LibriSpeech/dev-clean/2035/147961/2035-147961-0018.flac"))
# input_tensor = torch.randn(1, 1, 32000)  # Example with a single audio clip of length 32000
write("ExtraMics16/AudioTests/tnet1b.wav", 48000,angled_speech.T[0])
input_tensor=torch.from_numpy((angled_speech.T[0].T.reshape((1,1,-1))))
# input_tensor=input_tensor.astype(torch.float64)
input_tensor=input_tensor.float()
print(type(input_tensor))
# Forward pass through the model
output = model(input_tensor)


print("Output 1 shape:", output.shape)
# print("Output 2 shape:", output2.shape)
write("ExtraMics16/AudioTests/tnet1p.wav", 48000,(output.detach().numpy())[0][0])
write("ExtraMics16/AudioTests/tnet2p.wav", 48000,(output.detach().numpy())[0][1])
# write("ExtraMics16/AudioTests/tnet3p.wav", 48000,(output.detach().numpy())[0][2])