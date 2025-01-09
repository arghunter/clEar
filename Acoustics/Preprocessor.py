import numpy as np


class Preprocessor:
    # Mirroered: channels 1 and 2 are iodentical, 3 and 4 are identical and so on. Culls half the channels
    def __init__(self,mirrored=False,interpolate=1):
        self.mirrored = mirrored
        self.interpolate=interpolate
    def process(self,samples):
        
        if self.mirrored:
            arr=np.zeros((samples.shape[0],int(samples.shape[1]/2)))
            for i in range(samples.shape[0]):
                
                for j in range(len(arr[0])):
                    arr[i][j]=samples[i,j*2]
               
            samples=arr
        
        inter=np.zeros((samples.shape[0]*self.interpolate,samples.shape[1]))
        for i in range(samples.shape[0]-1):
            for j in range(samples.shape[1]):
                for k in range(self.interpolate):
                    inter[self.interpolate*i+k][j]=samples[i][j]+((samples[i+1][j]-samples[i][j])/self.interpolate)*k
        for k in range(self.interpolate):
            for j in range(samples.shape[1]):
                inter[self.interpolate*(samples.shape[0]-1)+k][j]= samples[(samples.shape[0]-1)][j]+((samples[samples.shape[0]-1][j]-samples[samples.shape[0]-2][j])/self.interpolate)*k
        samples=inter
        return samples
        
