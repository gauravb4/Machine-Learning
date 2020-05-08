import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    rootdir = "fma_small"
    imagedir = "fma_small_img"
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file
            if(filepath.endswith("mp3")):
                print(filepath)
                filename = file[:-4]
                # print(filename)
                # y, sr = librosa.load("fma_small\\000\\000002.mp3")
                y,sr = librosa.load(str(filepath))
                spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
                spect = librosa.power_to_db(spect, ref=np.max)

                fig = plt.figure(figsize=(2.5,1))
                librosa.display.specshow(spect, x_axis=None, y_axis=None, sr = sr, fmax=8000)
                plt.tight_layout()
                # plt.show()
                fig.savefig(imagedir + os.sep + filename + ".png")
                print(imagedir + os.sep + filename + ".png")
                count += 1



    print(count)


if __name__ == '__main__':
    main()