import dataset as d
from mel_spectogram import MelSpecto
from bounding_box import Bbox

import multiprocessing
from time import time
import os


def data_processing(file_name, sig, s_r, s, f_r_c, o_f):
    """

    :param file_name: file_name
    :param sig: processed_waveform
    :param s_r: resample_rate
    :param s: second
    :param f_r_c: filter_repetition_count
    :param o_f: output_folder
    :return:
    """
    fig_size = (10, 5)
    mel_f_n = file_name + '_' + str(f_r_c)
    mel_spec = MelSpecto(s_r, o_f, mel_f_n)
    mel_spec.image_save(sig, fig_size)

    bbox = Bbox(o_f, m_s=s)
    try:
        r = bbox.bbox(mel_f_n)
    except ValueError as e:
        print('Error', mel_f_n)
        print(e)
        return

    bbox.anno_save_file(r)
    bbox.labeled_save_image(r, fig_size)

    anno_data, image = bbox.cut_black(r, fig_size)
    bbox.labeled_save_image(anno_data, fig_size, image=image, sub_name='cropped')


if __name__ == '__main__':
    data_folder_path = '../../data_1107/'
    # data_folder_path = '../../2/'
    resample_rate = 8000
    second = 5
    filter_repetition_count = 5
    output_folder = f'../../data_mel_spec/spec_{resample_rate}_{second}/'
    # output_folder = f'../../2/'

    dataset = d.SoundData(data_folder_path)

    dataset.resample_rate = resample_rate
    dataset.max_second = second
    dataset.filter_repetition_count = filter_repetition_count
    dataset.output_folder = output_folder

    print('데이터셋 변경')
    print('시작', len(dataset.wav_files))
    data_set_list = list()
    start_time = time()
    for w_f in dataset.wav_files:
        data_set_list.append(dataset.make_csv(w_f))
    print(time() - start_time)
    print('데이터셋 전처리 완료')

    print(len(data_set_list))
    start_time = time()
    cpu_count = os.cpu_count()
    print(cpu_count)
    pool = multiprocessing.Pool(processes=cpu_count)

    output_list = pool.starmap(data_processing, data_set_list)
    # 프로세스 풀 종료
    pool.close()
    pool.join()
    print(time() - start_time)

    # for i in data_set_list:
    #     print(i)
    #     data_processing(*i)
    #     break


    # 50298_TV.wav