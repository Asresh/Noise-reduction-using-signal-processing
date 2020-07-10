import IPython
from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import io
%matplotlib inline

url = "https://raw.githubusercontent.com/bhanupratap1810/uni/master/fish.wav"
response = urllib.request.urlopen(url)
data, rate = sf.read(io.BytesIO(response.read()))
data = data

IPython.display.Audio(data=data, rate=rate)

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(data)

noise_len = 2 # seconds
noise = band_limited_noise(min_freq=2000, max_freq = 12000, samples=len(data), samplerate=rate)*10
noise_clip = noise[:rate*noise_len]
audio_clip_band_limited = data+noise

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(audio_clip_band_limited)

IPython.display.Audio(data=audio_clip_band_limited, rate=rate)

noise_reduced = nr.reduce_noise(audio_clip=audio_clip_band_limited, noise_clip=noise_clip, prop_decrease=1.0, verbose=True)

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(noise_reduced)

IPython.display.Audio(data=noise_reduced, rate=rate)

url = "https://raw.githubusercontent.com/bhanupratap1810/uni/master/noise_cafe.wav"
response = urllib.request.urlopen(url)
noise_data, noise_rate = sf.read(io.BytesIO(response.read()))

fig, ax = plt.subplots(figsize=(20,4))
ax.plot(noise_data)

IPython.display.Audio(data=noise_data, rate=noise_rate)

max(noise_data)

snr = 2 # signal to noise ratio
noise_clip = noise_data/snr
audio_clip_cafe = data + noise_clip


fig, ax = plt.subplots(figsize=(20,4))
ax.plot(audio_clip_cafe)

IPython.display.Audio(data=audio_clip_cafe, rate=noise_rate)

noise_reduced = nr.reduce_noise(audio_clip=audio_clip_cafe, noise_clip=noise_clip, verbose=True)

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(noise_reduced)

IPython.display.Audio(data=noise_reduced, rate=rate)

noise_reduced = nr.reduce_noise(audio_clip=audio_clip_cafe.astype('float32'),
                                noise_clip=noise_clip.astype('float32'),
                                use_tensorflow=True, 
                                verbose=False)


len(noise_reduced), len(audio_clip_cafe)

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(audio_clip_cafe)
ax.plot(noise_reduced, alpha = 0.5)


IPython.display.Audio(data=noise_reduced, rate=rate)
















