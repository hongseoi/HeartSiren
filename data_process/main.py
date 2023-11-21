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


if __name__ == '__main__':
    data_folder_path = '../../data_1107/'
    resample_rate = 20000
    second = 10
    filter_repetition_count = 5
    output_folder = f'spec_{resample_rate}_{second}/'

    dataset = d.SoundData(data_folder_path)

    dataset.resample_rate = resample_rate
    dataset.max_second = second
    dataset.filter_repetition_count = filter_repetition_count
    dataset.output_folder = output_folder

    print('데이터셋 변경')
    print('시작', len(dataset.wav_files))
    data_set_list = list()
    # 병렬 처리를 위한 map 메서드 호출
    # data_set_list = pool.starmap(dataset.make_csv, dataset.wav_files)
    # # 프로세스 풀 종료
    # pool.close()
    # pool.join()

    for w_f in dataset.wav_files:
        data_set_list.append(dataset.make_csv(w_f))

    print('데이터셋 전처리 완료')

    # data_set = dataset.data_pre_processing()
    print(len(data_set_list))
    # for idx, data in enumerate(dataset):
    #     file_name = data['file_name']
    #     sig = data['waveform']
    #
    #     mel_f_n = file_name + '_' + str(filter_repetition_count)
    #     mel_spec = MelSpecto(resample_rate, output_folder, mel_f_n)
    #     mel_spec.image_save(sig, fig_size)
    #
    #     bbox = Bbox(output_folder, m_s=second)
    #     try:
    #         r = bbox.bbox(mel_f_n)
    #     except ValueError:
    #         print('Error', idx, mel_f_n)
    #         continue
    #     bbox.anno_save_file(r)
    #     bbox.labeled_save_image(r, fig_size)
    start_time = time()
    cpu_count = os.cpu_count()
    print(cpu_count)
    pool = multiprocessing.Pool(processes=cpu_count)

    output_list = pool.starmap(data_processing, data_set_list)
    # 프로세스 풀 종료
    pool.close()
    pool.join()
    print(time() - start_time)
