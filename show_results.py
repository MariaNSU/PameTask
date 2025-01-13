import matplotlib.pyplot as plt
import numpy as np
from process_files import (get_data_files, find_max_temp_by_interpol, compute_activation_energy,
                           process_files, save_results_to_file)

def read_data(filename):
    experiments = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        experiment_name = None
        data = []
        for line in lines:
            if line == '\n':
                continue
            line = line.strip()
            if line.endswith('.dat'):
                if experiment_name and data:
                    experiments[experiment_name] = data
                experiment_name = line
                data = []
            else:
                values = list(map(float, line.split()))
                data.append(values)
        if experiment_name and data:
            experiments[experiment_name] = data
    return experiments


def plot_data(experiment_name, data):
    '''
    Помощник в отрисовке обработанных данных
    :param experiment_name: имя файла с экспериментов для подписи кривой
    :param data: сами обработанные данные
    :return:
    '''
    en = [row[0] for row in data]
    T = [row[1] for row in data]

    ln_en_T2 = [np.log(e / t ** 2) for e, t in zip(en, T)]
    inv_T = [1 / t for t in T]

    plt.plot(inv_T, ln_en_T2, marker='o', label=experiment_name)



def plot_ln_en_T2(filename='res.txt'):
    '''
    Рисует логарифм скорости эмиссии, нормированной на квадрат температуры, от обратной температуры
    :param filename: имя файла с результатами
    :return:
    '''
    experiments = read_data(filename)
    plt.figure(figsize=(12, 8))
    for experiment_name, data in experiments.items():
        plot_data(experiment_name, data)

    plt.xlabel('1/T')
    plt.ylabel('ln(en/T^2)')
    plt.title('Графики ln(en/T^2) от 1/T')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()




def plot_found_maximums():
    '''
    Рисует максимумы найденные с помощью интерпляции Гауссом. Найденные максимумы отмечены звездочкой того же цвета,
    что и сама кривая для данных
    :return:
    '''
    data_files = get_data_files()
    freqs = np.loadtxt('s049c\\freq.dat')

    plt.figure(figsize=(13, 8))
    data = np.loadtxt(data_files[2])
    T = data[:, 0]  # Первый столбец - температура
    conductance_data = data[:, 1:]  # Остальные столбцы - проводимость
    cmap = plt.cm.get_cmap('viridis', len(conductance_data.T))
    for i in range(len(conductance_data.T)):
        G = conductance_data.T[i]
        freq = freqs[i]
        max_params = find_max_temp_by_interpol(G, T)
        max_temp = max_params[0]
        max_conductance = max_params[1]
        plt.scatter(max_temp, max_conductance, c=cmap(i),  marker='*')
        plt.plot(T, G, marker='+', c=cmap(i), label=data_files[1].split('.')[0] + " " + str(freq))

    plt.xlabel('T, K')
    plt.ylabel('G/(2*pi*f), pF')
    plt.title('G/(2*pi*f) от T')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()


def print_activation_energy(filename='result.txt'):
    experiments = read_data(filename)
    for experiment_name, data in experiments.items():
        en = [row[0] for row in data]
        T = [row[1] for row in data]
        Ea = compute_activation_energy(np.array(T), np.array(en))[0]
        print(f"Ea = {Ea: .4f} eV for {experiment_name.split('\\')[1]}")


'''#  Пример использования функций
data_files = get_data_files()
result = process_files(data_files)
save_results_to_file(result)
'''
print_activation_energy()

