import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def tsv_load(file_path):
    f = open(file_path, 'r')
    data = f.read().splitlines()

    anno_dict = {'1': list(), '3': list()}
    for d in data:
        anno = d.split('\t')
        if anno[2] in ['1', '3']:
            anno_dict[anno[2]].append(anno[:2])
    return anno_dict


def plot_waveform(waveform, sample_rate, file_name=None, anno_dict=None, xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    # axes.axis('off')
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        # axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)

        if anno_dict:
            for anno, value in anno_dict.items():
                color = 'red'
                if anno == '3':
                    color = 'blue'
                for v in value:
                    v = list(map(float, v))
                    axes[c].add_patch(
                        patches.Rectangle(
                            (v[0], -1),  # (x, y)
                            v[1] - v[0], 2,  # width, height
                            edgecolor=color,
                            fill=False,
                        ))
    # figure.suptitle(title)
    # plt.show(block=False)

    plt.savefig(f'{file_name}_wave.png', bbox_inches="tight", pad_inches=0)


def plot_specgram(waveform, sample_rate, xlim=None, save=False, file_name=None, anno_dict=None):
    waveform = waveform.numpy()
    print(len(waveform[0]))
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    # axes.axis('off')

    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
          axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
          axes[c].set_xlim(xlim)

        if anno_dict:
            for anno, value in anno_dict.items():
                color = 'red'
                if anno == '3':
                    color = 'blue'
                for v in value:
                    v = list(map(float, v))
                    axes[c].add_patch(
                        patches.Rectangle(
                            (v[0], 0),  # (x, y)
                            v[1] - v[0], 2000,  # width, height
                            edgecolor=color,
                            fill=False,
                        ))

    # figure.suptitle(title)
    # plt.show(block=False)
    if save:
        save_folder = 'image'
        if anno_dict:
            save_folder = 'annotations'
        plt.savefig(f'{save_folder}/{file_name}_specgram.png', bbox_inches="tight", pad_inches=0)
        # plt.savefig(f'{file_name}_specgram.png')
        plt.close()