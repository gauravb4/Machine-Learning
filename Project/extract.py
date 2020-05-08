import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

def main():
    rootdir = "fma_small"
    imagedir = "fma_small_img"
    metadata = "tracks.csv"
    output_file = "genres_small.csv"

    tracks = pd.read_csv(metadata, index_col=0, header=[0,1])
    tracks = tracks['track']['genre_top']
    track_genres = {}
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file
            if(filepath.endswith("mp3")):
                print(filepath)
                filename = file[:-4]

                try:
                    y,sr = librosa.load(str(filepath))
                except:
                    print("Error loading: " + filename)
                else:
                    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
                    spect = librosa.power_to_db(spect, ref=np.max)

                    fig = plt.figure(figsize=(2.5,1), frameon=False)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    librosa.display.specshow(spect, x_axis=None, y_axis=None, sr = sr, fmax=8000, cmap='gray_r')
                    plt.tight_layout()
                    fig.savefig(imagedir + os.sep + filename + ".png", bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                    genre = tracks[int(filename)]
                    track_genres[filename] = genre
                    count += 1

    with open(output_file, 'w') as f:
        w = csv.writer(f)
        w.writerows(track_genres.items())
    print("Total Count: " + str(count))

if __name__ == '__main__':
    main()
