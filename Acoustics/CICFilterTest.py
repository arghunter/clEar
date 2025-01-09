import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write,read
import soundfile as sf
filename="./Acoustics/pdm_sinewave.txt"
import matplotlib
matplotlib.use("Qt5Agg")
def binary_to_decimal(binary_str):
    """
    Convert a two's complement binary string to a decimal integer.
    """
    if binary_str[0] == '1':  # negative number
        inverted_str = ''.join('1' if b == '0' else '0' for b in binary_str)
        return -1 * (int(inverted_str, 2) + 1)
    else:
        return int(binary_str, 2)

def decimal_to_binary(decimal, bits=19):
    """
    Convert a decimal integer to a two's complement binary string of 32 bits.
    """
    if decimal < 0:
        decimal = (1 << bits) + decimal
    binary_str = bin(decimal)[2:]  # convert to binary and remove '0b' prefix
    return binary_str.zfill(bits)

def twos_complement_addition(bin1, bin2, bits=19):
    """
    Perform two's complement binary addition with 32-bit padding.
    """
    # Ensure input binary strings are 32 bits
    bin1 = bin1.zfill(bits)
    bin2 = bin2.zfill(bits)

    dec1 = binary_to_decimal(bin1)
    dec2 = binary_to_decimal(bin2)
    result_decimal = dec1 + dec2

    # Wrap around if necessary
    max_value = 1 << (bits - 1)
    min_value = -1 * max_value
    if result_decimal >= max_value:
        result_decimal -= 2 * max_value
    elif result_decimal < min_value:
        result_decimal += 2 * max_value

    return decimal_to_binary(result_decimal, bits)
def binary_not(binary_str, bits=19):
    """
    Perform bitwise NOT operation on a binary string with 32-bit padding.
    """
    binary_str = binary_str.zfill(bits)
    return ''.join('1' if b == '0' else '0' for b in binary_str)

def twos_complement_subtraction(bin1, bin2, bits=19):
    """
    Perform two's complement binary subtraction with 32-bit padding.
    """
    # Perform bin1 - bin2 as bin1 + (-bin2)
    bin2_not = binary_not(bin2, bits)
    bin2_neg = twos_complement_addition(bin2_not, '1'.zfill(bits), bits)
    return twos_complement_addition(bin1, bin2_neg, bits)
# Example usage
# bin1 = "1101"  # -3 in 4-bit two's complement, will be padded to 32-bit
# bin2 = "1110"  # 6 in 4-bit two's complement, will be padded to 32-bit
# result_add = twos_complement_addition(bin1, bin2)
# result_sub = twos_complement_subtraction(bin1, bin2)
# print(f"Result of adding {bin1} and {bin2} in 32-bit two's complement: {result_add}")
# print(f"Result of subtracting {bin2} from {bin1} in 32-bit two's complement: {result_sub}")


def cic(arr):
    int1="0"
    int2="0"
    int3="0"
    dif1="0"
    dif2="0"
    dif3="0"
    out="0"
    counter=1
    out_array=[]
    in_array=[]
    count2=0

    for line in arr:
        # line=file.readline()

        
        count2+=1
        int1=twos_complement_addition(int1,decimal_to_binary(int(line)))
        in_array.append(decimal_to_binary(int(line)))
        int2=twos_complement_addition(int1,int2)
        int3=twos_complement_addition(int2,int3)
        if counter>=64:

            out=twos_complement_subtraction(twos_complement_subtraction(twos_complement_subtraction(int3,dif1),dif2),dif3)
            dif3=twos_complement_subtraction(twos_complement_subtraction(int3,dif1),dif2)

            dif2=twos_complement_subtraction(int3,dif1)
            dif1=int3
            counter=0
            out_array.append(binary_to_decimal(out))
            # print(binary_to_decimal(out))
        counter+=1;
            # line=file.readline()
    nparr=np.array(out_array, dtype=float)
    print(max(nparr))
    nparr/=max(nparr)
    return nparr
    # print(len(nparr))
    # write("cictest.wav", 48000,np.array(out_array))
    # plt.plot(nparr[0:2000])
    # plt.show()
    # nparr=np.array(in_array, dtype=float)
    # print(max(nparr))
    # nparr/=max(nparr)
    # plt.plot(nparr[0:2000])
    # plt.show()
# import scipy
# print(np.sum(np.square(nparr)))#
# write("ExtraMics16/AudioTests/cictest.wav", 48000,nparr)
# print("hereq1")
# plt.plot(nparr)

# Add labels and title for better understanding
# N =int(4096)
# # sample spacing
# T = 1.0 / 48000.0
# x = np.linspace(0.0, N*T, N)
# y = nparr
# yf = scipy.fftpack.fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

# fig, ax = plt.subplots()
# ax.plot(xf[0:300], (2.0/N * np.abs(yf[:N//2]))[0:300])
# plt.show()
# # Display the plot
# plt.show()
