#A program to create a digital filter for a second order Butterworth low pass filter
import math

#define an output array for the audio file post-LPF
y = []

#define variables for calculations
#sampling rate, fs, Hz
fs = 8820
#sampling period, T, s
T = 1.1133*(10**-4)
#cutoff frequencies, w, Hz
wd1 = 2*math.pi*300
wd2 = 2*math.pi*2300
#attenuation, A, dB
A = 30

#determine frequency response using prewarping
#analog frequency, wa, Hz
wa1 = (2/T)*math.tan((wd1*T)/2)
wa2 = (2/T)*math.tan((wd2*T)/2)

#find wa values
print(wa1)
print(wa2)

#n is the order of the filter, need to round up
n = (math.log10(A**2 - 1))/(2*math.log10(wa2/wa1))
#find n value
print(n)

#substitute in wa1 into the 2nd order Butterworth LPF equation using the bilinear transformation
#write the resultant difference equation


#take the inverse DFT of the data to convert it to an audio file and save
