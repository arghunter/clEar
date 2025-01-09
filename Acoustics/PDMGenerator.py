import numpy as np
import matplotlib.pyplot as plt

def generate_pdm_sinewave(frequency, pdm_rate, duration,delay, noise):
    """
    Generate a PDM (Pulse Density Modulation) sine wave.

    Parameters:
        frequency (float): Frequency of the sine wave in Hz.
        pdm_rate (int): Bit rate of the PDM stream in Hz.
        duration (float): Duration of the signal in seconds.

    Returns:
        np.ndarray: PDM bitstream (1s and 0s).
    """
    # Time vector for the sine wave
    t = np.linspace(0, duration, int(pdm_rate * duration), endpoint=False)

    # Generate sine wave
    sinewave = 0.8*np.sin(2 * np.pi * frequency * t)
    noise_shape = sinewave.shape 
    
    noise = np.random.normal(0, noise, noise_shape)
    
    sinewave = sinewave + noise
    sinewave=np.roll(sinewave,int(delay))
    # Normalize sine wave to range 0-1
    sinewave = (sinewave + 1) / 2

    # PDM encoding
    accumulator = 0.0
    pdm_stream = []
    for sample in sinewave:
        accumulator += sample  # Add the normalized sample
        if accumulator >= 0.5:  # Threshold for outputting a 1
            pdm_stream.append(1)
            accumulator -= 1.0
        else:
            pdm_stream.append(0)

    return np.array(pdm_stream, dtype=np.uint8)

def write_bitstream_to_text(bitstream, filename):
    """
    Write a PDM bitstream to a text file, one bit per line.

    Parameters:
        bitstream (np.ndarray): The PDM bitstream (1s and 0s).
        filename (str): The output file name.
    """
    with open(filename, 'w') as f:
        for bit in bitstream:
            f.write(f"{bit}\n")

# # Parameters
# frequency = 1000  # Sine wave frequency in Hz
# pdm_rate = 48000*64  # PDM bit rate in Hz
# duration = 1 # Duration of the sine wave in seconds
# output_file = './Acoustics/pdm_sinewave.txt'  # Output file name

# # Generate the PDM sine wave
# pdm_sinewave = generate_pdm_sinewave(frequency, pdm_rate, duration,50*64,0.2)

# # Write the PDM bitstream to a text file
# write_bitstream_to_text(pdm_sinewave, output_file)

# # Plot the PDM signal (first 1000 bits)
# plt.figure(figsize=(12, 4))
# plt.step(range(1000), pdm_sinewave[:1000], where='post')
# plt.title("PDM Sine Wave (First 1000 Bits)")
# plt.xlabel("Bit Index")
# plt.ylabel("PDM Bit Value")
# plt.grid()
# plt.show()

# print(f"PDM bitstream written to {output_file}")
