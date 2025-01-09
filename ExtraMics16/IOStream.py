from scipy.io.wavfile import read
import numpy as np
from queue import Queue
import sounddevice as sd
# from AudioWriter import AudioWriter
#wav to arr as j Samples,Channels
class IOStream: #sample duration in microseconds
    def __init__(self,frame_len=20000,frame_shift=10000):
        self.frame_len=frame_len
        self.frame_shift=frame_shift
        self.q=Queue()
        pass
    
    # Call one of the three below
    
    def arrToStream(self,arr,sample_rate):
        self.const=True
        self.arr=arr
        self.frequency=sample_rate
        if(len(self.arr.shape)>1):
            self.channels=arr.shape[1]
        else: 
            self.channels=1
        dur=(1/self.frequency)*10**6
        n_samples = int(self.frame_len/dur)
        sample_shift=int(self.frame_shift/dur)
        iter=0
        
        while iter < self.arr.shape[0]:
            self.q.put(self.arr[iter:iter+n_samples])
            iter+=sample_shift
        if (self.arr[iter: self.arr.shape[0]]).shape[0]>0:
            self.q.put(self.arr[iter: self.arr.shape[0]])
    def wavToStream(self,filename):
        self.const=True
        file = read(filename)
        self.arr = np.array(file[1])
        self.arrToStream(self.arr,file[0])
            
    def audio_callback(self,indata, frames, time, status):
        # print(("1"))
        for i in range(frames):
            # print('')
            if(self.buffer_head<self.buffer2.shape[0]):
                self.buffer2[self.buffer_head]=indata[i]
                self.buffer_head+=1
            else:
                self.q.put(np.concatenate([self.buffer1,self.buffer2]))
                self.buffer1=np.array(self.buffer2,copy=True)               
                self.buffer_head=0

        
        
    def streamAudio(self,frequency,channels):
       
        self.const=False
        self.frequency=frequency
        self.channels=channels
        dur=(1/self.frequency)*10**6
        n_samples = int(self.frame_shift/dur)
        self.buffer1=np.ones((n_samples,channels))
        self.buffer2=np.ones((n_samples,channels))
        self.buffer_head=0
        stream=sd.InputStream(
            device=None, channels=self.channels,
            samplerate=self.frequency, callback=self.audio_callback)
        stream.start()
        
    def complete(self):
        return self.const and self.q.empty()
    def getNextSample(self):
        
        return self.q.get()

# sd.default.device=18
# io= IOStream()
# io.streamAudio(48000,8)
# # io.wavToStream("./beamformingarray/AudioTests/test1.wav")
# writer= AudioWriter()
# while True:
#     print((io.getNextSample()))
    # writer.add_sample(io.getNextSample())
# print(io.arr.shape)
# print(writer.data.shape)
# writer.write("./beamformingarray/AudioTests/test20.wav",48000)   
