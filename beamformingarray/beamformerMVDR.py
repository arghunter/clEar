import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import scipy.signal as signal
from IOStream import IOStream
from AudioWriter import AudioWriter
from VAD import VAD
from MUSIC import MUSIC
import threading
import pickle
from DelayApproximation import DelayAproximator

C=343.3
class Beamformer():
    
    def __init__(self,sample_rate=48000,spacing=np.array([[0,0],[0.028,0],[0.056,0],[0.084,0],[0.112,0],[0.14,0],[0.168,0],[0.196,0]]),num_channels=8,exp_avg=50,frame_len=960,stft_len=1024 ,srctrck=0):
        self.spacing=spacing
        self.num_channels=num_channels
        self.exp_avg=exp_avg
        self.frame_count=1
        self.frame_len=frame_len
        self.stft_len=stft_len
        self.stftd2=int(stft_len/2)
        self.sample_rate=sample_rate
        self.frame_shift=int(frame_len/2)
        win=np.hanning(frame_len)
        win=win*self.frame_shift/sum(win)
        self.win_multi=np.tile(win,(self.num_channels,1)).T
        self.N_f=int(stft_len/2)+1
        self.mu=0
        self.global_covar=np.zeros((num_channels,num_channels,self.N_f),dtype='complex128')
        self.vad=VAD(16000)
        self.theta=-0.1
        self.speech=False
        self.MUSIC=MUSIC(spacing=spacing,num_channels=num_channels,srctrk=srctrck)
        self.c=6
        
        self.music_freq=10
        self.fail_count=0
        self.delay_approx=DelayAproximator(self.spacing)
        self.doalock=False
        # self.
        
    def beamform(self,frame):
        if(len(frame)!=self.frame_len):
            return np.zeros((self.frame_len,1))
        win_data = np.asmatrix(frame*self.win_multi)
        
        
        spectrum=np.asmatrix(np.fft.rfft(win_data,self.stft_len,axis=0))

        if self.frame_count < self.exp_avg:
            self.mu=(self.frame_count-1)/self.frame_count
        for k in range(0,self.N_f):
            cov_mat=np.dot(spectrum[k,:].T, np.conj(spectrum[k,:]))
            corr_mat=cov_mat/np.trace(cov_mat); 

            self.global_covar[:, :, k] = self.mu * self.global_covar[:, :, k] + (1 - self.mu) * corr_mat
    
        
        self.speech=self.vad.is_speech(win_data)

        
        if self.speech and self.doalock==False:
            if self.c>self.music_freq:
                covar=self.global_covar.copy()

                self.MUSIC.doa(covar)
                self.c=0
            self.c+=1
            self.theta=self.MUSIC.sources[self.MUSIC.nsrc-1]
            
            
            X=spectrum.T[0].T

            Y=spectrum.T[3].T
            R = np.multiply(X,np.conj(Y));
        
            tphat = np.real(np.fft.ifft(R/np.abs(R),axis=0));
            tphat=np.reshape(tphat,(-1))
            tphat=np.concatenate([tphat[int(len(X)/2):len(X)],tphat[0:int(len(X)/2)]])
            locs, _ = signal.find_peaks(tphat, height=None, distance=None)
            sorted_indices = np.argsort(tphat[locs])[::-1]
            pks = tphat[locs][sorted_indices]
            locs = locs[sorted_indices]
            dif=1/self.sample_rate*(locs[0]-len(X)/2)
            # dif=343.3*dif/6/0.028
            # print(locs[0]-512)
            dif=C*dif/(np.sqrt((self.spacing[0][0]-self.spacing[self.num_channels-1][0])**2+(self.spacing[0][1]-self.spacing[3][1])**2)) 
            # print(dif)
            if dif<-1:
                dif=-1
            if dif>1:
                dif=1
            ang=(np.degrees(np.arccos(dif))+360)%360
            # print("Angle:"+str(ang))
            # print("theta"+str(self.theta))

            if(ang!=0 and ang!=180 and (((np.abs(ang-self.theta)>75)and self.theta<180) or((np.abs(360-ang-self.theta)>75)and self.theta>180))):
                
                self.c=self.music_freq+1
            # if(ang!=0 and ang!=180):
            #     self.theta=ang
            
        if self.doalock:
            time=self.time
        else:
            time= np.asmatrix(self.delay_approx.get_delays(DelayAproximator.get_pos(self.theta,0.5)))
        w=np.asmatrix(np.zeros((self.num_channels,self.N_f),dtype='complex128'))
        for k in range (0,self.N_f-1):
            f=k*self.sample_rate/self.stft_len;
            alpha=np.exp(-1j*2*np.pi*f*time).T
            r_inv=np.linalg.pinv(self.global_covar[:,:,k]+(1e-8)*np.eye(self.num_channels)) 
            w[:,k]=r_inv@alpha/(np.conj(alpha.T)@r_inv@alpha)
        rec_signal=np.multiply(w.H,spectrum[0:self.N_f,:])

        submatrix = rec_signal[1:-1, :]

        flipped_conjugate = np.flipud(np.conj(submatrix))
        rec_signal = np.vstack([rec_signal, flipped_conjugate])


        summed_signal=(np.sum(rec_signal,axis=1))
        
        res_comp=(np.fft.ifft(summed_signal, axis=0))
        res=np.real(res_comp)
        # print(self.frame_count)
        res=res[0:self.frame_len]
        # print((output[i:i + frame_len, :]).shape)
        self.frame_count+=1
        return res
    def set_doa(self,doa):
        self.theta=doa
        self.doalock=True
        self.time = np.asmatrix(self.delay_approx.get_delays(DelayAproximator.get_pos(self.theta,0.5)))
    
