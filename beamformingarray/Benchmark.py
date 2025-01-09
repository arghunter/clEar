import numpy as np
from IOStream import IOStream
from beamformerMVDR import Beamformer
from AudioWriter import AudioWriter
from Preprocessor import Preprocessor
from time import time
from CrossCorrelator import CrossCorrelatior
from VAD import VAD
import soundfile as sf
from SignalGen import SignalGen
from Signal import Sine,Sawtooth,Chirp,Square
import Signal
import matplotlib.pyplot as plt
from pystoi import stoi
speech_database_path="C:\\Users\\arg\\Documents\\Datasets\\dev-clean.tar\\dev-clean\\LibriSpeech"
speech_database_subset="dev-clean"
speech_database_local=speech_database_path+"\\"+speech_database_subset
speakers=[]
chapters=[]
target_samplerate=48000
num_microphones=8
# np.array([[0,0],[0.028,0]])
spacings=[np.array([[0,0],[0.028,0],[0.056,0],[0.084,0]]),np.array([[0,0],[0.028,0],[0.056,0],[0.084,0],[0.112,0],[0.14,0],[0.168,0],[0.196,0]]),np.array([[-0.08,0.042],[-0.08,0.014],[-0.08,-0.014],[-0.08,-0.042],[0.08,0.042],[0.08,0.014],[0.08,-0.014],[0.08,-0.042]]),np.array([[-0.08,0.042],[-0.08,0.014],[-0.08,-0.028],[-0.08,-0.042],[0.08,0.042],[0.08,0.014],[0.08,-0.028],[0.08,-0.042]])]
def open_chapter(chapter_num):
    print(chapter_num)
    chapter_string=speech_database_local+"\\"+chapters[chapter_num][1]+"\\"+chapters[chapter_num][0]+"\\"
    chapter_trans=open(chapter_string+chapters[chapter_num][1]+"-"+chapters[chapter_num][0]+".trans.txt") 
    chapter_splits=[]
    while(True):
        line=chapter_trans.readline()
        if(len(line)<=0):
            break
        arr=line.split(" ")
        chapter_splits.append(arr)
        chapter_splits[len(chapter_splits)-1][0]=chapter_string+arr[0]+".flac"
    return chapter_splits




def read_speakers():
    speakers_file=open(speech_database_path+"\\SPEAKERS.TXT")
    while(True):
        line =speakers_file.readline()
        if(len(line)<=0):
            break
        if( line[0]!=';'):
            arr=line.split('|')
            if(speech_database_subset in arr[2] ):
                for i in range(len(arr)):
                    arr[i]=arr[i].strip()
                speakers.append(arr)
            
def read_chapters():
    chapters_file=open(speech_database_path+"\\CHAPTERS.TXT")
    while(True):
        line =chapters_file.readline()
        if(len(line)<=0):
            break
        if( line[0]!=';'):
            arr=line.split('|')
            for i in range(len(arr)):
                    arr[i]=arr[i].strip()
            if(speech_database_subset in arr[3]):
                chapters.append(arr)
def generate_angles(num):
    angles=[]
    for i in range(num):
        angles.append(int(np.random.rand()*360)) # IMPORTANT!!!!: THIS IS JUST FOR SINGLE LINEAR ARRAY NEED THE TRANFORMATION FOR 2 ARRAY SYSTEM

    return angles



##########################################################################################

Synthetic Signal Tests
num_single_source_synthetic_tests=0
synthetic_tests_length=5
min_frequency=100
max_frequency=1000
synthetic_single_source_rms=[]
synthetic_single_source_signed_mean=[]
synthetic_single_source_unsigned_mean=[]
synthetic_single_source_angles=generate_angles(num_single_source_synthetic_tests)
sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
synthetic_single_source_freq=[]
for i in range(num_single_source_synthetic_tests):
    print(i)
    doa_diff=[]
    beamformer=Beamformer(num_microphones,spacing,target_samplerate)
    freq=np.random.randint(min_frequency,max_frequency)
    synthetic_single_source_freq.append(freq)
    if(i+1==num_single_source_synthetic_tests):
        signal=Chirp(start_freq=min_frequency,end_freq=max_frequency,sample_rate=target_samplerate)
    elif(i%3==0):
        signal=Sine(frequency=freq,sample_rate=target_samplerate)
    elif(i%3==1):
        signal=Sawtooth(frequency=freq,sample_rate=target_samplerate)
    elif(i%3==2):
        signal=Square(frequency=freq,sample_rate=target_samplerate)
    
    sig_data=signal.generate_wave(synthetic_tests_length)
    sig_gen.update_delays(synthetic_single_source_angles[i])
    angled_sig_data=sig_gen.delay_and_gain(sig_data)
    io=IOStream()
    io.arrToStream(angled_sig_data,target_samplerate)
    aw=AudioWriter()
    while(not io.complete()):
        frame=io.getNextSample()
        aw.add_sample(beamformer.beamform(frame))
        doa_diff.append(synthetic_single_source_angles[i]-beamformer.doa)
        # print(doa_diff)
    
    doa_diff_arr=np.array(doa_diff)
    synthetic_single_source_rms.append(np.sqrt(np.mean(doa_diff_arr**2)))
    synthetic_single_source_signed_mean.append(np.mean(doa_diff_arr))
    synthetic_single_source_unsigned_mean.append(np.mean(np.abs(doa_diff_arr)))
print(np.mean(synthetic_single_source_unsigned_mean))
print(np.mean(synthetic_single_source_signed_mean))
print(np.mean(synthetic_single_source_rms))

#Single Source Attenuation Test
read_speakers()
read_chapters()
attenuation_test_length=0.5
min_angle=-180
max_angle=180
source_angle=0
min_frequency=3000
max_frequency=3000
sig_gen=SignalGen(num_microphones,spacings[3],target_samplerate)
chapter_splits=open_chapter(0)
speech,samplerate=sf.read(chapter_splits[0][0])
# print(chapter_splits[0][0])
speech=np.reshape(speech,(-1,1))
interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))

speech=interpolator.process(speech)
sig=Sine(frequency=min_frequency)
sig_data=speech[0:48000*4]
# sig_data=sig.generate_wave(0.5)
sig_gen.update_delays(source_angle)
angled_sig_data=sig_gen.delay_and_gain(sig_data)
noise=2/3*np.random.randn(*angled_sig_data.shape)
noise_angled_speech=noise+angled_sig_data
angles=[]
stoi_diff=[]
for i in range(min_angle,max_angle,5):
    
    io=IOStream()
    io.arrToStream(noise_angled_speech,target_samplerate)
    aw=AudioWriter()
    beamformer=Beamformer(num_channels=num_microphones,spacing=spacings[3])
    beamformer.set_doa(i)
    print(i)
    while(not io.complete()):
        frame=io.getNextSample()
        # print(frame)
        aw.add_sample(beamformer.beamform(frame),480)
        
        # print(doa_diff)

    aw.write("./beamformingarray/AudioTests/15.wav",48000)


 
    stoi_diff.append( stoi(sig_data, aw.data[0:len(sig_data)], target_samplerate, extended=False)-stoi(sig_data, noise_angled_speech.T[0].T[0:len(sig_data)].reshape(-1,1), target_samplerate, extended=False))




for i in stoi_diff:
    print(i)

plt.figure(figsize=(10,4),dpi=100)

plt.plot(angles_arr,rms_data_arr)
plt.grid(True)
# plt.set_xlim([-90, 90])
# plt.set_xlabel("Theta (Degrees)")
# plt.set_ylabel("Amplitude (Decibels)")
# plt.axvline(x = source_angle, color = 'b', label = 'axvline - full height')
print(rms_data_arr)
plt.show()

Multiple Source Attenuation Test
attenuation_test_length=0.5
min_angle=0
max_angle=180
source1_angle=70
distance1=1
source2_angle=110
distance2=1
min_frequency=100
max_frequency=3000
sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
# sig1=Chirp(min_frequency,max_frequency)
sig1=Chirp(start_freq=min_frequency,end_freq=max_frequency,sample_rate=target_samplerate)
sig1_data=sig1.generate_wave(attenuation_test_length)
sig_gen.update_delays(source1_angle)
sig_gen.update_gains(distance1)
angled_sig1_data=sig_gen.delay_and_gain(sig1_data)
sig2=Sine(frequency=max_frequency/2+min_frequency/2,sample_rate=target_samplerate)
sig2_data=sig2.generate_wave(attenuation_test_length)
sig_gen.update_delays(source2_angle)
sig_gen.update_gains(distance2)
angled_sig2_data=sig_gen.delay_and_gain(sig2_data)
summed_data=Signal.sum_signals(angled_sig1_data,angled_sig2_data)

angles=[]
rms_data=[]
for i in range(min_angle,max_angle):
    print(i)
    angles.append(i)
    beamformer=Beamformer(num_microphones,spacing,target_samplerate)
    beamformer.toggle_doa_lock()
    beamformer.update_delays(i)
    attenuated_sig_data=beamformer.beamform(summed_data)
    rms=np.sqrt(np.mean(attenuated_sig_data**2))
    rms_data.append(rms)    
angles_arr=np.array(angles)
rms_data_arr=(np.array(rms_data))
rms_data_arr=rms_data_arr/max(rms_data_arr)
plt.figure(figsize=(10,4),dpi=100)
ax = plt.subplot(131)
ax.plot(angles_arr,rms_data_arr)
ax.grid(True)
ax.set_xlim([0, 180])
ax.set_xlabel("Theta (Degrees)")
ax.set_ylabel("Amplitude (Decibels)")
plt.axvline(x = source1_angle, color = 'b', label = 'axvline - full height')
plt.axvline(x = source2_angle, color = 'r', label = 'axvline - full height')
plt.show()

# Single Source Tests
read_speakers()
read_chapters()
print('=============================Single Source Everything tests===================================')

num_single_source_tests=10
single_source_angles=generate_angles(num_single_source_tests)# IMPORTANT!!!!: THIS IS JUST FOR SINGLE LINEAR ARRAY NEED THE TRANFORMATION FOR 2 ARRAY SYSTEM
# single_source_angles[0]=24
single_source_noises=np.random.rand(num_single_source_tests) *2/3
# single_source_noises[0]=0.8
for spacing in spacings:
    spacing=spacings[3]
    num_microphones=len(spacing)
    
    ang_iter=0
    chapter_ind=0
    chapter_splits=open_chapter(chapter_ind)
    split_iterator=0
    interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
    sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
    single_source_rms=[]
    single_source_signed_mean=[]
    single_source_unsigned_mean=[]
    single_source_signal_mse=[]
    single_source_org_mse=[]
    single_source_snr=[]
    single_source_stoi_org=[]
    
    stoiarr=[]
    print(single_source_angles)
    for i in range(num_single_source_tests):
        # print(i)
        doa_diff=[]
        beamformer=Beamformer(num_channels=num_microphones,spacing=spacing)
        if chapter_ind>=len(chapters):
            chapter_ind=0
        if split_iterator>=len(chapter_splits):
            split_iterator=0
            chapter_ind+=1
            chapter_splits=open_chapter(chapter_ind)
        speech,samplerate=sf.read(chapter_splits[split_iterator][0])
        speech=np.reshape(speech,(-1,1))
        speech=interpolator.process(speech)
        power_clean_signal = np.sqrt(np.mean(np.square(speech*32767)))
        sig_gen.update_delays(single_source_angles[i])
        angled_speech=sig_gen.delay_and_gain(speech)
        noise=single_source_noises[i]*np.random.randn(*angled_speech.shape)
        noise_angled_speech=angled_speech+noise
        power_noise=np.sqrt(np.mean(np.square((noise.T[0][0:len(speech)]) *32767)))
        io=IOStream()
        io.arrToStream(noise_angled_speech,target_samplerate)
        aw=AudioWriter()
        print(single_source_angles[i])
        while(not io.complete()):
            frame=io.getNextSample()
            # print(frame)
            aw.add_sample(beamformer.beamform(frame),480)
            if(beamformer.speech and beamformer.theta!=180):
                doa_diff.append(single_source_angles[i]-(beamformer.theta))
            # print(doa_diff)
        split_iterator+=1
        aw.write("./beamformingarray/AudioTests/15.wav",48000)
        print(transcribe_from_numpy(aw.data.reshape((-1))))
        SNR_db1=20 * np.log10(power_clean_signal / power_noise)
        power_cleaned = np.sqrt(np.mean(np.square((aw.data[0:len(speech)]-speech) *32767)))
        SNR_db2 = 20 * np.log10(power_clean_signal / power_cleaned)
        single_source_snr.append(SNR_db1-SNR_db2)
        doa_diff_arr=np.array(doa_diff)
        single_source_stoi_org.append(stoi(speech, noise_angled_speech.T[0].T[0:len(speech)].reshape(-1,1),48000));
        stoiarr.append( stoi(speech, aw.data[0:len(speech)], target_samplerate, extended=False)-stoi(speech, noise_angled_speech.T[0].T[0:len(speech)].reshape(-1,1), target_samplerate, extended=False))
        print(transcribe_from_numpy(aw.data.reshape((-1))))
        single_source_org_mse.append(np.square(np.subtract(org_data,op_data)).mean() )
        single_source_signal_mse.append(np.square(np.subtract(org_data,noise_data)).mean() )
        single_source_rms.append(np.sqrt(np.mean(doa_diff_arr**2)))
        single_source_signed_mean.append(np.mean(doa_diff_arr))
        single_source_unsigned_mean.append(np.mean(np.abs(doa_diff_arr)))
    print(spacing)
    print("Avg unsigned mean doa:" + str(np.mean(single_source_unsigned_mean)))
    print("Avg signed mean doa:"+str(np.mean(single_source_signed_mean)))
    print("Avg rms mean:"+str(np.mean(single_source_rms)))
    print("Avg stoi diff mean:"+str(np.mean(stoiarr)))
    print(single_source_angles)
    print(single_source_noises)
    print(single_source_unsigned_mean)
    print(stoiarr)
    print(single_source_snr)
    print(single_source_stoi_org)
    print(np.mean(single_source_stoi_org))
    
print('=============================Multi Source Source Tracking Tests===================================')
num_multi_source_tests=0
multi_source_angles=generate_angles(num_multi_source_tests+1)# IMPORTANT!!!!: THIS IS JUST FOR SINGLE LINEAR ARRAY NEED THE TRANFORMATION FOR 2 ARRAY SYSTEM
# multi_source_angles[0]=24
multi_source_noises=np.random.rand(num_multi_source_tests+1)
for spacing in spacings:
    # spacing=spacings[3]
    num_microphones=len(spacing)
    
    ang_iter=0
    chapter_ind=0
    chapter_splits=open_chapter(chapter_ind)
    split_iterator=0
    interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
    sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
    multi_source_rms=[]
    multi_source_signed_mean=[]
    multi_source_unsigned_mean=[]
    multi_source_signal_mse=[]
    multi_source_org_mse=[]
    multi_source_opp_doa_err=[]
    # person detection in noise envs
    stoiarr=[]
    print(multi_source_angles)
    for i in range(num_multi_source_tests):
        # print(i)
        doa_diff=[]
        opp_diff=[]
        beamformer=Beamformer(num_channels=num_microphones,spacing=spacing,srctrck=1)
        if chapter_ind>=len(chapters):
            chapter_ind=0
        if split_iterator>=len(chapter_splits):
            split_iterator=0
            chapter_ind+=1
            chapter_splits=open_chapter(chapter_ind)
        speech,samplerate=sf.read(chapter_splits[split_iterator][0])
        if split_iterator>=len(chapter_splits)-1:
            opp,samplerate=sf.read(chapter_splits[0][0])
        else:
            opp,samplerate=sf.read(chapter_splits[split_iterator+1][0])
        speech=np.reshape(speech,(-1,1))
        opp=0.75*np.reshape(opp,(-1,1))
        if len(opp)>len(speech):
            opp=opp[0:len(speech)]
        speech=interpolator.process(speech)
        opp=interpolator.process(opp)
        sig_gen.update_delays(multi_source_angles[i])
        angled_speech=sig_gen.delay_and_gain(speech)
        sig_gen.update_delays(multi_source_angles[i+1])
        angled_opp=sig_gen.delay_and_gain(opp)
        angled_speech=Signal.sum_signals(angled_speech,angled_opp)
        noise_angled_speech=angled_speech+multi_source_noises[i]*np.random.randn(*angled_speech.shape)
        # print(noise_angled_speech.shape)
        aw=AudioWriter()
        aw.add_sample(noise_angled_speech,0)
        aw.write("./beamformingarray/AudioTests/16.wav",48000)
        io=IOStream()
        io.arrToStream(noise_angled_speech,target_samplerate)
        aw=AudioWriter()
        # print(multi_source_angles[i])
        while(not io.complete()):
            frame=io.getNextSample()
            # print(frame)
            aw.add_sample(beamformer.beamform(frame),480)
            if(beamformer.speech and beamformer.theta!=180):
                # print(beamformer.MUSIC.sources)
                doa_diff.append(min(np.abs(multi_source_angles[i]-(beamformer.MUSIC.sources[0])),min((np.abs(multi_source_angles[i]-(beamformer.MUSIC.sources[1]))),(np.abs(multi_source_angles[i]-(beamformer.MUSIC.sources[2]))))))
                opp_diff.append(min(np.abs(multi_source_angles[i+1]-(beamformer.MUSIC.sources[0])),min((np.abs(multi_source_angles[i+1]-(beamformer.MUSIC.sources[1]))),(np.abs(multi_source_angles[i+1]-(beamformer.MUSIC.sources[2]))))))
            # print(doa_diff)
        split_iterator+=1
        aw.write("./beamformingarray/AudioTests/15.wav",48000)
        doa_diff_arr=np.array(doa_diff)
        stoiarr.append( stoi(speech, aw.data[0:len(speech)], target_samplerate, extended=False)-stoi(speech, noise_angled_speech.T[0].T[0:len(speech)].reshape(-1,1), target_samplerate, extended=False))
        
        # multi_source_org_mse.append(np.square(np.subtract(org_data,op_data)).mean() )
        # multi_source_signal_mse.append(np.square(np.subtract(org_data,noise_data)).mean() )
        multi_source_rms.append(np.sqrt(np.mean(doa_diff_arr**2)))
        multi_source_signed_mean.append(np.mean(doa_diff_arr))
        multi_source_unsigned_mean.append(np.mean(np.abs(doa_diff_arr)))
        multi_source_opp_doa_err.append(np.mean(opp_diff))
    print(spacing)
    print("Avg unsigned mean doa:" + str(np.mean(multi_source_unsigned_mean)))
    print("Avg stoi diff mean:"+str(np.mean(stoiarr)))
    print(multi_source_angles)
    print(multi_source_noises)
    print(multi_source_unsigned_mean)
    print(stoiarr)
    print(multi_source_opp_doa_err)
    
    

print('=============================Multi Source Stoi SNR tests===================================')
num_multi_source_tests=6
multi_source_angles=generate_angles(num_multi_source_tests+1)# IMPORTANT!!!!: THIS IS JUST FOR SINGLE LINEAR ARRAY NEED THE TRANFORMATION FOR 2 ARRAY SYSTEM
# multi_source_angles[0]=24
# multi_source_angles[1]=120
multi_source_noises=np.random.rand(num_multi_source_tests+1)
# multi_source_noises[0]=0.1
for spacing in spacings:
    spacing=spacings[3]
    num_microphones=len(spacing)
    
    ang_iter=0
    chapter_ind=0
    chapter_splits=open_chapter(chapter_ind)
    split_iterator=0
    interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
    sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
    multi_source_rms=[]
    multi_source_signed_mean=[]
    multi_source_unsigned_mean=[]
    multi_source_signal_mse=[]
    multi_source_org_mse=[]
    multi_source_snr=[]
    multi_source_stoi_org=[]
    # person detection in noise envs
    stoiarr=[]
    print(multi_source_angles)
    for i in range(num_multi_source_tests):
        # print(i)
        doa_diff=[]
        beamformer=Beamformer(num_channels=num_microphones,spacing=spacing,srctrck=1)
        if chapter_ind>=len(chapters):
            chapter_ind=0
        if split_iterator>=len(chapter_splits):
            split_iterator=0
            chapter_ind+=1
            chapter_splits=open_chapter(chapter_ind)
        speech,samplerate=sf.read(chapter_splits[split_iterator][0])
        if split_iterator>=len(chapter_splits)-1:
            opp,samplerate=sf.read(chapter_splits[0][0])
        else:
            opp,samplerate=sf.read(chapter_splits[split_iterator+1][0])
        speech=np.reshape(speech,(-1,1))
        opp=0.75*np.reshape(opp,(-1,1))
        # beamformer.set_doa(multi_source_angles[i])
        if len(opp)>len(speech):
            opp=opp[0:len(speech)]
        speech=interpolator.process(speech)
        power_clean_signal = np.sqrt(np.mean(np.square(speech*32767)))
        
        opp=interpolator.process(opp)
        sig_gen.update_delays(multi_source_angles[i])
        angled_speech=sig_gen.delay_and_gain(speech)
        sig_gen.update_delays(multi_source_angles[i+1])
        angled_opp=sig_gen.delay_and_gain(opp)
        angled_speech=Signal.sum_signals(angled_speech,angled_opp)
        noise=multi_source_noises[i]*np.random.randn(*angled_speech.shape)
        noise_angled_speech=angled_speech+noise
        # print(opp.shape)
        power_noise=np.sqrt(np.mean(np.square((opp.reshape((-1))+noise.T[0][0:min(len(speech),len(opp))]) *32767)))
        aw=AudioWriter()
        aw.add_sample(noise_angled_speech,0)
        aw.write("./beamformingarray/AudioTests/16.wav",48000)
        io=IOStream()
        io.arrToStream(noise_angled_speech,target_samplerate)
        aw=AudioWriter()
        # print(multi_source_angles[i])
        while(not io.complete()):
            frame=io.getNextSample()
            # print(frame)
            aw.add_sample(beamformer.beamform(frame),480)
            if(beamformer.speech and beamformer.theta!=180):
                # print(beamformer.MUSIC.sources)
                doa_diff.append(multi_source_angles[i]-(beamformer.theta))
            # print(doa_diff)
        split_iterator+=1
        aw.write("./beamformingarray/AudioTests/15.wav",48000)
        doa_diff_arr=np.array(doa_diff)
        multi_source_stoi_org.append(stoi(speech, noise_angled_speech.T[0].T[0:len(speech)].reshape(-1,1), target_samplerate, extended=False))
        stoiarr.append( stoi(speech, aw.data[0:len(speech)], target_samplerate, extended=False)-stoi(speech, noise_angled_speech.T[0].T[0:len(speech)].reshape(-1,1), target_samplerate, extended=False))
        SNR_db1=20 * np.log10(power_clean_signal / power_noise)
        # print(SNR_db)
        power_cleaned = np.sqrt(np.mean(np.square((aw.data[0:len(speech)]-speech) *32767)))
        SNR_db2 = 20 * np.log10(power_clean_signal / power_cleaned)
        # print(SNR_db)
        multi_source_snr.append(SNR_db1-SNR_db2)
        # multi_source_org_mse.append(np.square(np.subtract(org_data,op_data)).mean() )
        # multi_source_signal_mse.append(np.square(np.subtract(org_data,noise_data)).mean() )
        multi_source_rms.append(np.sqrt(np.mean(doa_diff_arr**2)))
        multi_source_signed_mean.append(np.mean(doa_diff_arr))
        multi_source_unsigned_mean.append(np.mean(np.abs(doa_diff_arr)))
    print(spacing)

    print("Avg stoi diff mean:"+str(np.mean(stoiarr)))

    print(multi_source_noises)
    print(multi_source_unsigned_mean)
    print(stoiarr)
    print(multi_source_snr)
    print(multi_source_stoi_org)
    # print(multi)
print(np.mean(single_source_signal_mse))
print(np.mean(single_source_org_mse))
num_multi_source_tests=1
multi_source_angles=generate_angles(num_multi_source_tests+1)# IMPORTANT!!!!: THIS IS JUST FOR SINGLE LINEAR ARRAY NEED THE TRANFORMATION FOR 2 ARRAY SYSTEM
# multi_source_angles[0]=24
multi_source_noises=np.random.rand(num_multi_source_tests+1)*2/3 *0
for spacing in spacings:
    spacing=spacings[3]
    num_microphones=len(spacing)
    
    ang_iter=0
    chapter_ind=0
    chapter_splits=open_chapter(chapter_ind)
    split_iterator=0
    interpolator=Preprocessor(mirrored=False,interpolate=int(np.ceil(target_samplerate/16000)))
    sig_gen=SignalGen(num_microphones,spacing,target_samplerate)
    multi_source_rms=[]
    multi_source_signed_mean=[]
    multi_source_unsigned_mean=[]
    multi_source_signal_mse=[]
    multi_source_org_mse=[]

    # person detection in noise envs
    stoiarr=[]
    print(multi_source_angles)
    for i in range(num_multi_source_tests):
        # print(i)
        doa_diff=[]
        beamformer=Beamformer(num_channels=num_microphones,spacing=spacing,srctrck=1)
        speech=Sine(500,1).generate_wave(4)
        power_clean_signal = np.mean(np.square(speech))
        opp=Sine(1234,0.75).generate_wave(4)
        sig_gen.update_delays(multi_source_angles[i])
        angled_speech=sig_gen.delay_and_gain(speech)
        sig_gen.update_delays(multi_source_angles[i+1])
        angled_opp=sig_gen.delay_and_gain(opp)
        angled_speech=Signal.sum_signals(angled_speech,angled_opp)
        noise_angled_speech=angled_speech+multi_source_noises[i]*np.random.randn(*angled_speech.shape)
        print(noise_angled_speech.shape)
        # power_noisy_signal = np.mean(np.square(noise_angled_speech.T[0].T))
        # power_noise = np.mean(np.square(noise_angled_speech.T[0].T - speech))
        aw=AudioWriter()
        aw.add_sample(noise_angled_speech,0)
        aw.write("./beamformingarray/AudioTests/16.wav",48000)
        io=IOStream()
        io.arrToStream(noise_angled_speech,target_samplerate)
        aw=AudioWriter()
        # print(multi_source_angles[i])
        while(not io.complete()):
            frame=io.getNextSample()
            # print(frame)
            aw.add_sample(beamformer.beamform(frame),480)
            if(beamformer.speech and beamformer.theta!=180):
                # print(beamformer.MUSIC.sources)
                doa_diff.append(multi_source_angles[i]-(beamformer.theta))
            # print(doa_diff)
        split_iterator+=1
        aw.write("./beamformingarray/AudioTests/15.wav",48000)
        doa_diff_arr=np.array(doa_diff)
        stoiarr.append( stoi(speech, aw.data[0:len(speech)], target_samplerate, extended=False)-stoi(speech, noise_angled_speech.T[0].T[0:len(speech)].reshape(-1,1), target_samplerate, extended=False))
        power_noise = np.mean(np.square(aw.data[0:len(speech)] - speech))
        SNR_db = 10 * np.log10(power_clean_signal / power_noise)
        print(SNR_db)
        # multi_source_org_mse.append(np.square(np.subtract(org_data,op_data)).mean() )
        # multi_source_signal_mse.append(np.square(np.subtract(org_data,noise_data)).mean() )
        multi_source_rms.append(np.sqrt(np.mean(doa_diff_arr**2)))
        multi_source_signed_mean.append(np.mean(doa_diff_arr))
        multi_source_unsigned_mean.append(np.mean(np.abs(doa_diff_arr)))
    print(spacing)

    print("Avg stoi diff mean:"+str(np.mean(stoiarr)))

    print(multi_source_noises)
    print(multi_source_unsigned_mean)
    print(stoiarr)
    



