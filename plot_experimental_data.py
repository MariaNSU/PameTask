import numpy as np
import matplotlib.pyplot as plt
import os

def get_data_files(directory='s049c', sample_prefix='s049c'):
    """
    Получает список файлов, содержащих данные для образца с заданным префиксом.
    directory: str
        Директория, в которой находятся файлы с данными.
    sample_prefix: str
        Префикс имени файла, с которого начинаются файлы данных для конкретного образца.
    """
    data_files = []

    for file_name in os.listdir(directory):
        if file_name.startswith(sample_prefix) and file_name.endswith('.dat'):
            data_files.append(os.path.join(directory, file_name))

    return data_files


def plot_raw_data(file_num=1):
    '''
    Рисует экспериментальные данные из файла с номером file_num.
    :return:
    '''
    data_files = get_data_files()
    freqs = np.loadtxt('s049c\\freq.dat')

    plt.figure(figsize=(13, 8))
    file = data_files[file_num]  # Отрисуем экспериментальные данные из первого файла
    data = np.loadtxt(file)
    T = data[:, 0]  # Первый столбец - температура
    conductance_data = data[:, 1:]  # Остальные столбцы - проводимость
    for i in range(len(conductance_data.T)):
        G = conductance_data.T[i]
        freq = freqs[i]
        plt.plot(T, G, marker='o', label=file.split('.')[0] + " " + str(freq))

    plt.xlabel('T, K')
    plt.ylabel('G/(2*pi*f), pF')
    plt.title(f'G/(2*pi*f) от T для эксперимента {file.split('\\')[1]}')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
