# Importing libaries
import os
import numpy as np
import librosa as librosa
import librosa.display
from matplotlib import pyplot as plt

# Path where sample audios are located.
audio_path = r'./audios'
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


def convert_to_spectrogram(amp, sr, name, dst_folder='./images2/'):
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

def convert_to_spectrogram2(amp, sr, name, dst_folder='./images3/'):
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
    # Crop image.
    plt.tight_layout()
    plt.axis('off')

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
        amp, sr = librosa.load(os.path.join(audio_path, f'{file}'), sr=44100)
        # Convert to spectrogram.
        convert_to_spectrogram2(amp, sr=44100, name=file[:-4])
        print('Spectrogram created.')


if __name__ == '__main__':
    main()