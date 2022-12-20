# Importing libaries
import os
import numpy as np
import librosa as librosa
import librosa.display
import noisereduce as nr 
from pydub import AudioSegment as am
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt

# Path where sample audios are located.
audio_path = r'./samples'
imgs_path = r'./images'


def create_save_path(dst_folder='audios/'):
    """
    Create a folder to save a file, in case it doesn't exist.

    Parameters
    ------------
    dst_folder : string, optional
            Destination folder where files will be saved.
    """
    if not os.path.exists(os.path.dirname(dst_folder)):
        os.makedirs(os.path.dirname(dst_folder))


def butterworth_filter(data, cut_freq, fs, order=9, ftype='low'):
    """
    Apply the Butterworth low-pass filter to certain data.

    Parameters
    -------------
    data : array_like
            Data to be filtered.
    cut_freq : array_like
            The critical frequency or frequencies. For lowpass and highpass 
            filters, Wn is a scalar; for bandpass and bandstop filters, 
            Wn is a length-2 sequence.
    fs : int
            Sample frequency of the signal.
    order : int, optional
            Filter's order to be applied.
    ftype : string, optional
            Filter's type. Default 'lowpass'.

    Return
    -------------
    lfilter : array_like
            The output of the digital filter.
    """
    b, a = butter(order, cut_freq, fs=fs, btype=ftype)
    return lfilter(b, a, data)


def process_audio(file, freq=44100, dst_folder='audios/'):
    """
    Using AudioSegment class from Pydub library, open a .wav file and resample it to 44100Hz. 
    The output is a one-channel audio (mono).

    Parameters
    ----------
    audio : string
            Path (including extension) where the audio is located.
    freq : int
            Desired frequency.
    dst_folder :  string, optional
            Destination path where the audio will be saved.

    Return
    ----------
    new_audio : string
            New file name of the converted audio.
    """
    extension = os.path.basename(file).split('.')[-1]
    # Open audio file.
    if extension == 'mp3':
        audio = am.from_file(file, format='mp3')
    elif extension == 'ogg':
        audio = am.from_file(file, format='ogg')
    else:
        audio = am.from_file(file, format='wav')
    # Resample to 16000Hz and set channels to one.
    audio = audio.set_frame_rate(freq).set_channels(1)        
    # Path where the audio will be saved.
    new_audio_path = '{dst}{name}_mono.wav'.format(dst=dst_folder, name=file.split('/')[-1][:-4])
    # Export output and save it.
    try:
        audio.export(new_audio_path, format='wav')
    except FileNotFoundError:
        create_save_path()
        audio.export(new_audio_path, format='wav')
    return new_audio_path, audio


def filter_noise(file):
    """
    Apply a low-pass filter to the audio to remove noise.
    
    Parameters
    -------------
    file : string
            Path where the audio is located.
            
    Return
    -------------
    filtered_audio : array_like
            The output of the digital filter.
"""
    amp, sr = librosa.load(file, sr=44100)
    # Compute the time vector.
    duration = len(amp)/sr
    time = np.arange(0, duration, 1/sr)
    reduced_noise = nr.reduce_noise(y=amp, sr=sr) # Noise reduction
    full_sound, phase = librosa.magphase(librosa.stft(reduced_noise))
    filtered_sound = librosa.decompose.nn_filter(full_sound,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
    filtered_sound = np.minimum(full_sound, filtered_sound)
    # Reconstruct the phase
    margin_i, margin_v = 2, 10
    power = 2
    mask_i = librosa.util.softmask(filtered_sound,
                                margin_i * (full_sound - filtered_sound),
                                power=power)

    mask_v = librosa.util.softmask(full_sound - filtered_sound,
                                margin_v * filtered_sound,
                                power=power)
    #  Once we have the masks, simply multiply them with the input spectrum
    sound_background = mask_i * full_sound   
    # Reconstruction of the phase
    return librosa.istft(sound_background*phase)   


def convert_to_spectrogram(amp, sr, name, dst_folder='./images/'):
    # Size of the Fast Fourier Transform (FFT), which will also be used as the window length
    n_fft=1024
    # Step or stride between windows. If the step is smaller than the window length, the windows will overlap
    hop_length=320
    # Specify the window type for FFT/STFT
    window_type ='hann'
    # Calculate the spectrogram as the square of the complex magnitude of the STFT
    mel_bins = 64 # Number of mel bands
    # Compute the Mel spectrogram.
    Mel_spectrogram = librosa.feature.melspectrogram(y=amp, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window_type, n_mels = mel_bins, power=2.0)
    mel_spectrogram_db = librosa.power_to_db(Mel_spectrogram, ref=np.max)
    # Plot MEL spectrogram.
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel',hop_length=hop_length)
    # Add format to the image displayed.
    plt.colorbar(format='%+2.0f dB')
    # Crop image.
    plt.tight_layout()
    # Save image.
    try:
        plt.savefig(f'{dst_folder}{name}.png', bbox_inches = 'tight', pad_inches = 0)
    except FileNotFoundError:
        create_save_path(dst_folder)
        plt.savefig(f'{dst_folder}{name}.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()


def main():
    # Get all files in the folder.
    files = [file for file in os.listdir(audio_path) if file[-4:] in ['.wav', '.mp3', '.ogg']]
    # Iterate over all files.
    for file in files:
        print(f'Processing `{file}` ...')
        # Get the path of the file.
        file_path = os.path.join(audio_path, file)
        # Process the audio.
        new_audio_path, audio = process_audio(file_path, freq=44100)
        print('Audio processed.')
        # Filter noise.
        filtered_audio = filter_noise(new_audio_path)
        print('Noise filtered.')
        # Apply a butterworth filter.
        CUT_FREQ = [250, 2500] # From 250 to 2500 Hz.
        filtered_amp = butterworth_filter(filtered_audio, CUT_FREQ, fs=44100, order=5, ftype='band')
        print('Butterworth filter applied.')
        # Convert to spectrogram.
        convert_to_spectrogram(filtered_amp, sr=44100, name=file[:-4])
        print('Spectrogram created.')


if __name__ == '__main__':
    main()