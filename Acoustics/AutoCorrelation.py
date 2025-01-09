#==============================================================================
#  Copyright (c) 2014-2018 Joel de Guzman. All rights reserved.
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#==============================================================================
from matplotlib.pyplot import figure, show
from numpy import arange, sin, pi
import numpy as np
length=1024
dec_rate=64
count=0
f = open("C:\\Users\\arg\\Documents\\MATLAB\\pdm_sine_wave_11.txt", "r")
data=[]
ssum=0
for x in f:
    c=int(x)
   #  print(ssum)
    if(c==-1):
        c=0;
    if(count>=dec_rate-1):
      if(ssum>=dec_rate/2):
         data.append(1)
      else:
         data.append(-1)
      count=0
      ssum=0
    else:
      ssum+=c
    count+=1
print(len(data))
data=data[0:length]
f = open("C:\\Users\\arg\\Documents\\MATLAB\\pdm_sine_wave_10.txt", "r")
data2=[]
count=0
ssum=0
for x in f:
    c=int(x)
    if(c==-1):
        c=0;
    if(count>=dec_rate-1):
      if(ssum>=dec_rate/2):
         data2.append(1)
      else:
         data2.append(-1)
      count=0
      ssum=0
    else:
      ssum+=c
    count+=1
data2=data2[0:length]
t = arange(0, length)
# input = 0.3*sin(2*pi*t) + 0.4*sin(4*pi*t) + 0.3*sin(6*pi*t)
input = np.array(data)
fig = figure(1)

ax1 = fig.add_subplot(311)
ax1.plot(t, data)
ax1.grid(True)
# ax1.set_ylim((-0.1, 1.1))

class zero_cross:
   def __init__(self):
      self.y = 0

   def __call__(self, s):
      if s < -0.1:
         self.y = 0
      elif s > 0.1:
         self.y = 1
      return self.y

# zc = zero_cross()
# trig = [zc(s) for s in input]
trig=input
ax2 = fig.add_subplot(312)
ax2.plot(t, data2)
ax2.grid(True)
ax2.set_ylim((-0.1, 1.1))

def count_ones(l):
   r = 0
   
   for e in l:
      if e:
         r += 1
   return r

cross = np.roll(data2,0)
results = []
len = int(len(trig)/2)
for i in range((len)):
   print(i)
   x = [a ^ b for a, b in zip(trig[0:len], cross[i:i+len])]
   results.append(count_ones(x))
results.extend(results)

ax3 = fig.add_subplot(313)
ax3.plot(results)
ax3.grid(True)
# ax3.set_ylim((-5, 3000))

min_index = results[1:int(length/2-50)].index(min(results[1:int(length/2-50)]))
print((3.072*10**6)/((min_index+1)*dec_rate))
print(min_index+1)

show()


