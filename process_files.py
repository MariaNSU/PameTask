import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
import os


def gauss(x, A, x0, sigma):
    '''
    Функция для Гауссовой интерполяции
    :param x:
    :param A:
    :param x0:
    :param sigma:
    :return:
    '''
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def compute_activation_energy(temp, rates):
    '''
    Функция для энергии активации
    :param temp: максимальные температуры
    :param rates: скорость эмиссии
    :return:
    '''
    # Преобразуем данные: ln(en/T^2) → 1/T
    ln_en_T2 = np.log(rates / temp ** 2)
    inv_temp = 1 / temp
    # Линейная аппроксимация: ln(en/T^2) = -Ea/k * (1/T) + const
    popt = np.polyfit(inv_temp, ln_en_T2, 1, full=False)  # Наклон пропорционален -Ea/k
    slope = popt[0]

    k_boltzmann = 8.617333262145e-5  # эВ/К
    activation_energy = -slope * k_boltzmann  # Ea = -slope * k

    const = popt[1]
    return activation_energy, const
def complex_parametres_for_interpol(G, T):
    '''
    Интерполирует проводимость Гауссианом. Более сложный подбор параметров
    :param G:
    :param T:
    :return:
    '''
    # 1. Найти пики
    peaks, properties = find_peaks(G, height=5,distance=5)  # height - фильтрация по высоте, distance - минимальное расстояние между пиками

    # 2. Найти ширину пиков (опционально для фильтрации шумов)
    widths = peak_widths(G, peaks, rel_height=0.5)  # rel_height=0.5 - ширина на половине высоты

    # 3. Выбрать средний пик
    if len(peaks) > 1:
        mid_point = np.mean(T)  # Средняя точка диапазона по X
        closest_peak_index = np.argmin(np.abs(T[peaks] - mid_point))  # Найти ближайший пик к центру
    elif len(peaks) == 1:
        closest_peak_index = 0
    else:
        return []
    central_peak = peaks[closest_peak_index]
    peak_width = int(widths[0][closest_peak_index])  # Примерная ширина пика
    # 4. Получить данные вокруг центрального пика
    T_peak = T[max(0, central_peak - peak_width):min(len(T), central_peak + peak_width + 1)]
    G_peak = G[max(0, central_peak - peak_width):min(len(G), central_peak + peak_width + 1)]

    # Число параметров равно 3, значит длина массивов T и G не может быть меньше 3
    if len(G_peak) < 3 or len(T_peak) < 3:
        T_peak = T[central_peak:]
        G_peak = G[central_peak:]

    # 5. Интерполяция гауссом
    try:
        p0 = [max(G_peak), T[central_peak], 10]  # Начальные параметры: [амплитуда, центр, ширина]
        popt = curve_fit(gauss, T_peak, G_peak, p0=p0)
        return popt[0]

    except RuntimeError or TypeError:
        # Если не вышло интерполировать и более сложным способом, просто вернем температуру и проводиомость центрального пика
        return [T[central_peak], G[central_peak]]



def simple_parametres_for_interpol(conductance, temperatures):
    '''
    Интерполирует проводимость Гауссианом. Более простой подбор параметров
    :param G:
    :param T:
    :return:
    '''
    peaks, _ = find_peaks(conductance)  # Ищем локальные максимумы
    if len(peaks) > 0:
        peak_temps = temperatures[peaks]
        peak_conductance = conductance[peaks]
        # Интерполяция для одного максимума
        if len(peak_temps) > 1:
            # Если больше одного максимума, берем центральный
            peak_index = len(peak_temps) // 2
        else:
            peak_index = 0
        # Параметры для интерполяции
        initial_guess = (peak_conductance[peak_index], peak_temps[peak_index], 5)  # A, x0, sigma
        try:
            popt = curve_fit(gauss, temperatures, conductance, p0=initial_guess, maxfev=1500)
            return popt[0]
        except RuntimeError or TypeError as e:
            return []

def find_max_temp_by_interpol(conductance, temperatures):
    '''
    Поиск максимальной температуры по интерполированным данным
    :param conductance:
    :param temperatures:
    :return:
    '''
    # Гауссова интерполяция
    gauss_params = simple_parametres_for_interpol(conductance, temperatures)
    if len(gauss_params) == 0:
        gauss_params = complex_parametres_for_interpol(conductance, temperatures)
        # Сложный алгоритм не справился
        if len(gauss_params) == 0:
            return [-1, -1]
        # Сложный алгоритм не справился с интерполяцией, но есть центральный пик
        if len(gauss_params) == 2:
            T_central_peak = gauss_params[0]
            G_central_peak = gauss_params[1]
            return [T_central_peak, G_central_peak]

    if len(gauss_params) == 3:
        #temp_interp = np.linspace(min(temperatures), max(temperatures), 300)
        temp_interp = temperatures
        conductance_interp = gauss(temp_interp, gauss_params[0], gauss_params[1], gauss_params[2])

        # Фиксируем положение максимума после интерполяции
        max_conductance = np.max(conductance_interp)
        max_temp = temp_interp[np.argmax(conductance_interp)]

        if max_temp > 0:

            return [max_temp, max_conductance]
        else:
            return [-1, -1]


# Основная функция обработки данных
def process_files(data_files, freq_file='s049c/freq.dat'):
    '''
    Получает данные из файлов и применяет к ним интерполяцию и поиск максимумов
    :param data_files: список фалов с данными
    :param freq_file: файл с частотами
    :return:
    '''
    frequencies = np.loadtxt(freq_file)
    results = {}
    for data_file in data_files:
        data = np.loadtxt(data_file)
        temperatures = data[:, 0]  # Первый столбец - температура
        conductance_data = data[:, 1:]  # Остальные столбцы - проводимость
        max_temps = {}

        for i, conductance in enumerate(conductance_data.T):
            # Гауссова интерполяция
            max_temp = find_max_temp_by_interpol(conductance, temperatures)[0]
            if max_temp > 0:
                max_temps[frequencies[i]] = max_temp
            else:
                print("Interpolation error with: ", data_file, "at conductivity column: ", i)

        results[data_file] = max_temps
    return results

def save_results_to_file(res, output_file='result.txt'):
    '''
    Созраняет полученные результаты в файл
    :param res:
    :param output_file:
    :return:
    '''
    with open(output_file, "w") as f_out:
        for item in res.items():
            f_out.write(item[0] + "\n")
            for freq in item[1].items():
                temp = freq[1]
                en = freq[0] * np.pi  # en(Tmax) ~ w/2 ~ 2*pi*f/2

                f_out.write(str(en) + "\t" + str(temp) + "\n")
            f_out.write("\n")


def get_data_files(directory='s049c', sample_prefix='s049c'):
    '''
    Создает список файлов из директории с определенным префиксом
    :param directory:
    :param sample_prefix:
    :return:
    '''
    data_files = []
    for file_name in os.listdir(directory):
        if file_name.startswith(sample_prefix) and file_name.endswith('.dat'):
            data_files.append(os.path.join(directory, file_name))

    return data_files


'''
#  Пример использования функций  
data_files = get_data_files()
result = process_files(data_files)
save_results_to_file(result)
'''

