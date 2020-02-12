"""
Прототип программного обеспечения для управления манипулятором для автоградуировки
"""

import cv2
import random
import math
import os
import sqlite3
import serial
import json
import time
import numpy as np
import configparser
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.filters as filters
import serial.tools.list_ports as port_list
from tqdm import tqdm
from skimage import io
from skimage.color import rgb2gray
from PIL import Image, ImageOps
from io import BytesIO
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras.utils import np_utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense


# игнорируем предупреждения
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class ImageProcessing:
    """
    Общий класс для модулей получения и обработки изображения с камеры (либо изображения с HDD),
    а также для определения границ мер и захвата изображений цифр толщин мер
    """
    def __init__(self):
        """
        Инициализация экземпляра класса и определение переменных
        """
        self.opencv_image = None  # переменная для записи обработанного фильтрами изображения
        self.input_image = None  # переменная для записи оригинального изображения
        self.ADD = False  # переменная для отключения отдельных функциональностей модуля
        self.data_list = []  # переменная для записи обрезанных изображений, а также значений центров мер
        self.rectangle_area = (90000, 95000)  # кортеж площадей для фильтрации прямоугольников мер
        self.circle_area = (10000, 100000)  # кортеж площадей для фильтрации окружностей условного начала координат
        self.angle_area = (3000, 5000)  # кортеж площадей для фильтрации прямоугольников надписей толщин мер
        self.digit_area = (400, 550)  # кортеж площадей для фильтрации прямоугольников границ цифр толщин мер

    def image_tuning(self, image_path: str):
        """
        Метод для обработки входного изображения, а в случае его отсутствия захвата кадра с камеры, применения фильтра
        для выделения границ, сохранения отредактированного изображения в буфер обмена, настройки параметров HSV
        цветового фильтра

        :param image_path: путь к файлу изображения
        :return: None
        """
        # определение источника изображения
        if not os.path.exists(image_path):
            cap = cv2.VideoCapture(0)
            ret, img = cap.read()
            self.input_image = img
        else:
            self.input_image = io.imread(image_path)
        grayscale_image = rgb2gray(self.input_image)

        # уменьшение шумов на изображении вдоль границ объектов
        image_threshold = ((grayscale_image -
                            filters.threshold_local(grayscale_image, block_size=21,
                                                    method='gaussian'))*255).astype(np.uint8)

        # запись и выгрузка отфильтрованного изображения в/из оперативной памяти
        im = Image.fromarray(image_threshold)
        temp = BytesIO()
        im.save(temp, format='png')
        pil_image = Image.open(temp)

        numpy_image = np.array(pil_image)
        self.opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        # создание окна настроек начального и конечного цвета фильтра
        cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("settings")
        cv2.createTrackbar('h1', 'settings', 0, 255, self.nothing)
        cv2.createTrackbar('s1', 'settings', 0, 255, self.nothing)
        cv2.createTrackbar('v1', 'settings', 0, 255, self.nothing)
        cv2.createTrackbar('h2', 'settings', 255, 255, self.nothing)
        cv2.createTrackbar('s2', 'settings', 255, 255, self.nothing)
        cv2.createTrackbar('v2', 'settings', 255, 255, self.nothing)

        # считывание значений бегунков
        while True:
            hsv = cv2.cvtColor(self.opencv_image, cv2.COLOR_BGR2HSV)
            h1 = cv2.getTrackbarPos('h1', 'settings')
            s1 = cv2.getTrackbarPos('s1', 'settings')
            v1 = cv2.getTrackbarPos('v1', 'settings')
            h2 = cv2.getTrackbarPos('h2', 'settings')
            s2 = cv2.getTrackbarPos('s2', 'settings')
            v2 = cv2.getTrackbarPos('v2', 'settings')

            # формирование начального и конечного цвета фильтра
            h_min = np.array((h1, s1, v1), np.uint8)
            h_max = np.array((h2, s2, v2), np.uint8)

            # накладывание фильтра на кадр в цветовой модели HSV
            thresh = cv2.inRange(hsv, h_min, h_max)
            cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('result', thresh)
            ch = cv2.waitKey(5)
            if ch == 27:
                break
        cv2.destroyAllWindows()

        # создание файла настроек для сохранения в нем настроенных выше параметров HSV
        if not os.path.exists('settings.ini'):
            Config.create()
        config = Config()
        config.set_image_hsv_settings([h1, s1, v1, h2, s2, v2])
        config.change_settings()

    def borders_detection(self):
        """
        Метод для захвата границ мер, определения их центров, определения определение точки условного начала координат,
        выравнивания изображения в случае наклона (но не более 90 градусов в сторону) и захвата изображений цифр толщин мер

        :return: None
        """

        # загрузка параметров цветового фильтра из файла настроек
        config = Config()
        hsv_settings = config.get_image_tune_settings()
        hsv_min = np.array((hsv_settings[0], hsv_settings[1], hsv_settings[2]), np.uint8)
        hsv_max = np.array((hsv_settings[3], hsv_settings[4], hsv_settings[5]), np.uint8)

        # создание базы данных
        data_base = DataBase()
        data_base.create('data.db')

        # перевод изображения в цветовую модель HSV
        hsv = cv2.cvtColor(self.opencv_image, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, hsv_min, hsv_max)

        # определение контуров мер
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            center = (int(rect[0][0]), int(rect[0][1]))  # определение координат центра прямоугольника
            area = int(rect[1][0] * rect[1][1])
            if self.rectangle_area[0] < area < self.rectangle_area[1]:  # фильтруем лишние прямоугольники
                cv2.drawContours(self.opencv_image, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольники (для преднастройки)
                x, y, w, h = cv2.boundingRect(box)

                # запись в перменную обрезанных изображений (отфильтрованных и оригинальных) по границам мер, а также центров мер
                self.data_list.append([[self.opencv_image[y:y+h, x:x+w]], [self.input_image[y:y+h, x:x+w]],
                                       [center[0], center[1]]])

        # вывод отфильтрованных прямоугольников в кадр для преднастройки
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', self.opencv_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # поиск и фильтрование круглых объектов на изображении (кронштейн манипулятора) для условного начала координат
        if self.ADD:
            gray = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
            ret, gray_threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(gray_threshed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if 300 < len(cnt) < 500:
                    x_y, radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x_y[0]), int(x_y[1]))  # определяем координаты центра окружности
                    radius = int(radius)
                    area = math.pi*radius**2
                    if self.circle_area[0] < area < self.circle_area[1]:  # фильтруем лишние окружности
                        cv2.circle(self.opencv_image, center, radius, (255, 0, 0), 3)  # рисуем окружности (для преднастройки)

                        # запись в базу данных значения координат относительного начала координат
                        data_base.add_data('Начало координат', str(center[0]), str(center[1]))

            # вывод отфильтрованных окружностей в кадр для преднастройки
            cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            cv2.imshow('result', self.opencv_image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        # выравнивание изображений надписей толщин мер относительно горизонтали
        if not self.ADD:
            for i in range(len(self.data_list)):
                img_resh = self.data_list[i][0][0]
                img_orig = self.data_list[i][1][0]
                hsv = cv2.cvtColor(img_resh, cv2.COLOR_BGR2HSV)
                thresh = cv2.inRange(hsv, hsv_min, hsv_max)
                contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    center = (int(rect[0][0]), int(rect[0][1]))
                    area = int(rect[1][0] * rect[1][1])

                    # определение угла наклона прямоугольника к горизонтали
                    edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
                    edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))
                    used_edge = edge1
                    if cv2.norm(edge2) > cv2.norm(edge1):
                        used_edge = edge2
                    reference = (1, 0)
                    angle = 180.0 / math.pi * math.acos((reference[0] * used_edge[0] + reference[1] * used_edge[1]) /
                                                        (cv2.norm(reference) * cv2.norm(used_edge)))
                    if self.angle_area[0] < area < self.angle_area[1] and len(cnt) < 500:  # фильтр площадей прямоугольников
                        cv2.drawContours(img_resh, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольники (для преднастройки)
                        if angle > 90:
                            angle = 180 - angle
                        else:
                            angle = - angle
                        (h, w, d) = img_orig.shape

                        # поворот изображений мер и запись их в переменную
                        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated_resh = cv2.warpAffine(img_resh, rot, (w, h))
                        rotated_orig = cv2.warpAffine(img_orig, rot, (w, h))
                        self.data_list[i][0][0] = rotated_resh
                        self.data_list[i][1][0] = rotated_orig

                # вывод отфильтрованных прямоугольников в кадр для преднастройки
                cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('result', img_resh)
                cv2.waitKey()
                cv2.destroyAllWindows()

        # захват изображений цифр толщин мер
        if not self.ADD:
            for i in range(len(self.data_list)):
                img_resh = self.data_list[i][0][0]
                img_orig = self.data_list[i][1][0]
                hsv = cv2.cvtColor(img_resh, cv2.COLOR_BGR2HSV)
                thresh = cv2.inRange(hsv, hsv_min, hsv_max)
                contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    area = int(rect[1][0] * rect[1][1])
                    x, y, w, h = cv2.boundingRect(box)
                    if self.digit_area[0] < area < self.digit_area[1] and 0.6 < w/h < 0.8 and 20 < len(cnt) < 85:  # фильтрация прямоугольников
                        cv2.drawContours(self.data_list[i][0][0], [box], 0, (255, 0, 0), 2)  # рисуем прямоугольники (для преднастройки)
                        x, y, w, h = cv2.boundingRect(box)

                        # записываем захваченные изображения цифр в переменную
                        self.data_list[i].append([img_orig[y:y+h, x:x+w]])

                # вывод отфильтрованных прямоугольников в кадр для преднастройки
                cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('result', img_resh)
                cv2.waitKey()
                cv2.destroyAllWindows()

        # выгрузка изображений цифр из переменной, подача их в нейросеть для распознавания,
        # запись результатов распознования и координат центров мер в базу данных
        if not self.ADD:
            for i in range(len(self.data_list)):
                measure_thickness = str()
                for j in self.data_list[i][3:]:
                    net = DigitNN()
                    digit = net.use('keras_mnist.h5', j[0])
                    measure_thickness += str(digit)
                data_base.add_data(f'Мера №{i}', str(self.data_list[i][2][0]),
                                   str(self.data_list[i][2][1]), measure_thickness)

    @staticmethod
    def nothing(*arg):
        """
        Функция, которая необходима для вывода и работы окна с настройками HSV

        :param arg: *arg
        :return: None
        """
        pass


class Config:
    """
    Класс для создания и управления файлом конфигурации
    """
    def __init__(self):
        """
        Инициализация класса для создания файла конфигурации
        """
        config = configparser.ConfigParser()
        config.read('settings.ini')
        self.image_h1_settings = config.get('Settings', 'image_h1_settings')
        self.image_s1_settings = config.get('Settings', 'image_s1_settings')
        self.image_v1_settings = config.get('Settings', 'image_v1_settings')
        self.image_h2_settings = config.get('Settings', 'image_h2_settings')
        self.image_s2_settings = config.get('Settings', 'image_s2_settings')
        self.image_v2_settings = config.get('Settings', 'image_v2_settings')

    @staticmethod
    def create():
        """
        Метод для создания файла конфигурации

        :return: None
        """
        config = configparser.ConfigParser()
        config.add_section('Settings')
        config.set('Settings', 'image_h1_settings', '')
        config.set('Settings', 'image_s1_settings', '')
        config.set('Settings', 'image_v1_settings', '')
        config.set('Settings', 'image_h2_settings', '')
        config.set('Settings', 'image_s2_settings', '')
        config.set('Settings', 'image_v2_settings', '')
        with open('settings.ini', 'w') as config_file:
            config.write(config_file)

    def get_image_tune_settings(self):
        """
        Метод для выгрузки из файла конфигурации значений HSV цветового фильтра

        :return: list[self.image_HSV_settings]
        """
        return [self.image_h1_settings, self.image_s1_settings, self.image_v1_settings, self.image_h2_settings,
                self.image_s2_settings, self.image_v2_settings]

    def set_image_hsv_settings(self, image_hsv_settings: list):
        """
        Метод для записи значений HSV цветового фильтра в файл переменные

        :param image_hsv_settings: значения HSV цветового фильтра
        :return: nothing
        """
        self.image_h1_settings = image_hsv_settings[0]
        self.image_s1_settings = image_hsv_settings[1]
        self.image_v1_settings = image_hsv_settings[2]
        self.image_h2_settings = image_hsv_settings[3]
        self.image_s2_settings = image_hsv_settings[4]
        self.image_v2_settings = image_hsv_settings[5]

    def change_settings(self):
        """
        Метод для записи сделанных изменений в файл конфигурации

        :return: None
        """
        config = configparser.ConfigParser()
        config.read('settings.ini')
        config.set('Settings', 'image_h1_settings', str(self.image_h1_settings))
        config.set('Settings', 'image_s1_settings', str(self.image_s1_settings))
        config.set('Settings', 'image_v1_settings', str(self.image_v1_settings))
        config.set('Settings', 'image_h2_settings', str(self.image_h2_settings))
        config.set('Settings', 'image_s2_settings', str(self.image_s2_settings))
        config.set('Settings', 'image_v2_settings', str(self.image_v2_settings))
        with open('settings.ini', 'w') as config_file:
            config.write(config_file)


class DataBase:
    """
    Класс для создания и управления базой данных
    """
    def __init__(self):
        """
        Инициализация экземпляра класса
        """
        self.db_name = None

    def create(self, db_name: str):
        """
        Метод для создания базы данных формата SQLite3

        :param db_name: название базы данных
        :return: None
        """
        self.db_name = str(db_name)
        if not os.path.exists(db_name):
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute("""CREATE TABLE data (Data_name text, X_coord text, Y_coord text, Measure_thickness text)""")
            conn.commit()
            conn.close()
        return

    def add_data(self, data_name: str, x_coord: str, y_coord: str, thickness: str = None):
        """
        Метод для добавления данных в базу

        :param data_name: Наименование данных
        :param x_coord: х-координата
        :param y_coord: у-координата
        :param thickness: значение толщины меры
        :return: None
        """
        data = [(data_name), (x_coord), (y_coord), (thickness)]
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO data VALUES (?,?,?,?)", data)
        conn.commit()
        conn.close()

    def clear_db(self):
        """
        Метод для очистки базы данных

        :return: None
        """
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM data")
        conn.commit()
        conn.close()


class DigitNN:
    """
    Класс для создания и управления структурой сверточной нейросети для распознавания цифр на изображениях
    """
    def __init__(self):
        """
        Инициализация экземпляра класса и определение переменных
        """
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.color_limit = 160  # переменная для настройки порога усиления цвета

    def compile(self):
        """
        Метод для компиляции структуры сверточной нейросети

        :return: None
        """

        # Фиксируем seed для повторяемости результатов
        random.seed(0)
        np.random.seed(0)
        np.random.seed(0)

        # Размер изображения
        img_rows, img_cols = 28, 28

        # Загружаем данные
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        # Преобразование размерности изображений
        self.X_train = self.X_train.reshape(self.X_train.shape[0], img_rows, img_cols, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        # Нормализация данных
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255

        # Преобразование меток в категории
        self.y_train = np_utils.to_categorical(self.y_train, 10)
        self.y_test = np_utils.to_categorical(self.y_test, 10)

        # Создание последовательной модели
        self.model = Sequential()

        self.model.add(Conv2D(75, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(100, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(500, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

        # Компиляция модели
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def train_n_save(self, output_name: str):
        """
        Метод для обучения и сохранения нейросети

        :param output_name: название сохраняемой нейросети
        :return: None
        """
        # Обучение сети
        self.model.fit(self.X_train, self.y_train, batch_size=100, epochs=25, validation_split=0.2, verbose=2)

        # Оценка качества обучения сети на тестовых данных
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Точность на тестовых данных: {scores[1]*100}%")

        # Сохранение сети
        self.model.save(output_name)

    def use(self, nn_name: str, inp_image):
        """
        Метод для использования нейросети

        :param nn_name: название используемой нейросети
        :param inp_image: массив входного изображения
        :return: neuron_number
        """
        # загрузка модели нейросети
        model = tf.keras.models.load_model(nn_name)

        # определение структуры массива изображения для подачи в сеть
        data = np.ndarray(shape=(1, 28, 28, 1), dtype=np.float32)

        # перегон массива в PIL-формат для транформации в 1-канальное изображение и инверсии цветов
        img = Image.fromarray(inp_image)
        img = img.convert('L')
        img = ImageOps.invert(img)

        # изменение размера изображения и улучшение качества
        size = (28, 28)
        img = img.resize(size, Image.ANTIALIAS)

        # перегон изображения обратно в массив и усиление цвета
        image_array = np.asarray(img)
        img = image_array.copy()
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j] > self.color_limit:
                    img[i][j] = 255
                else:
                    img[i][j] = 0
        plt.imshow(img, 'gray')
        plt.show()
        # нормализация изображения, добавление размерности и передача в ннейросеть
        img = img.astype('float32')
        img /= 255.
        img = img.reshape((28, 28, 1))
        data[0] = img

        # загрузка в нейросеть
        prediction = model.predict(data)

        # получение номера нейрона с максимальным значением
        neuron_number = None
        for i in range(prediction.shape[1]):
            if prediction[0][i] == prediction[0].max().item():
                neuron_number = i
        print(neuron_number)
        return neuron_number


class VerticalControlNN:
    """
    Класс для определения момента соприкосновения измерительного датчика с поверхностью меры
    """
    def __init__(self):
        """
        Инициализация экземпляра класса и определение расположения данных для обучения и проверки
        """
        self.train_dir = 'train'  # определение директории для тренировочных данных
        self.val_dir = 'val'  # определение директории для валидационных данных
        self.test_dir = 'test'  # определение директории для тестовых данных
        self.img_width = 150  # задание размеров изображения
        self.img_height = 150  # задание размеров изображения
        self.input_shape = (self.img_width, self.img_height, 3)  # задание размера массива изображения
        self.epochs = 20  # количество тренировочных эпох
        self.batch_size = 1  # размер подвыборки
        self.nb_train_samples = 13  # количество изображений для обучения
        self.nb_validation_samples = 1  # количество изображений для валидации
        self.nb_test_samples = 1  # количество изображений для тестирования
        self.model = None

        # создание генератора изображений
        self.datagen = ImageDataGenerator(rescale=1. / 255)

        # создание генератора данных для обучения на основе изображений из каталога
        self.train_generator = self.datagen.flow_from_directory(self.train_dir,
                                                                target_size=(self.img_width, self.img_height),
                                                                batch_size=self.batch_size,
                                                                class_mode='binary')

        # создание генератора данных для проверки на основе изображений из каталога
        self.val_generator = self.datagen.flow_from_directory(self.val_dir,
                                                              target_size=(self.img_width, self.img_height),
                                                              class_mode='binary')

        # создание генератора данных для тестирования на основе изображений из каталога
        self.test_generator = self.datagen.flow_from_directory(self.test_dir,
                                                               target_size=(self.img_width, self.img_height),
                                                               batch_size=self.batch_size,
                                                               class_mode='binary')

    def compile(self):
        """
        Метод для компиляции структуры сверточной нейросети
        :return: None
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_n_save(self, output_name: str):
        """
        Метод для обучения и сохранения нейросети
        :param output_name: название сохраняемой нейросети
        :return: None
        """
        self.model.fit_generator(self.train_generator, steps_per_epoch=self.nb_train_samples // self.batch_size,
                                 epochs=self.epochs, validation_data=self.val_generator,
                                 validation_steps=self.nb_validation_samples // self.batch_size)
        self.model.save(output_name)

    @staticmethod
    def use(nn_name: str, inp_img: str):
        """
        Метод для использования нейросети
        :param nn_name: название используемой нейросети
        :param inp_img: массив входного изображения
        :return: 0 - датчик касается поверхности, 1 - датчик не касается поверхности
        """
        model = tf.keras.models.load_model(nn_name)
        data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
        if not os.path.exists(inp_img):
            cap = cv2.VideoCapture(0)
            ret, img = cap.read()
            image_array = img
        else:
            pic = Image.open(inp_img)
            pic = ImageOps.fit(pic, (150, 150), Image.LANCZOS)
            image_array = np.asarray(pic)
        img = image_array.astype('float32')
        img /= 255.
        data[0] = img
        prediction = model.predict(data)
        return int(round(prediction[0][0]))


class ManipulatorControl:
    """
    Класс для управления манипулятором
    """

    @staticmethod
    def get_probe_tester_com_port():
        """
        Метод для определения рабочего порта
        :return: port_name
        """
        port_name = None
        choose_port = []
        for port in port_list.comports():
            if 'STMicroelectronics Virtual COM Port' in port[1]:
                port_name = port[0]
                break
        return port_name

    @staticmethod
    def work(db_name: str):
        """
        Метод для непосредственного управления манипулятором
        :param db_name: название базы данных
        :return: None
        """
        filename = db_name
        sqlite3.register_adapter(np.int64, lambda val: int(val))
        conn = sqlite3.connect(filename, detect_types=sqlite3.PARSE_DECLTYPES)

        e = ExperimentalData.Experiment(conn, 8)

        h = 4

        charge = False
        with serial.Serial(ManipulatorControl.get_probe_tester_com_port(), 115200) as s:
            for i in tqdm(range(int(3600 * h))):
                d = str(s.readline())
                d = d[d.index('{'):1 + d.index('}')]
                if len(d) < 2:
                    continue
                if d[0] != '{' and d[-1] != '}':
                    continue
                d = json.loads(d)
                print(d)

                e.add_value(d['U'], d['I'])

                if not charge:
                    if d['U'] < 3400:
                        s.write('c')
                        charge = True
                if charge:
                    if d['U'] > 4100:
                        conn.close()
                        break

                time.sleep(1)

        with serial.Serial(ManipulatorControl.get_probe_tester_com_port(), 115200) as s:
            s.write(b'9')

        e.plot()


if __name__ == '__main__':
    image = ImageProcessing()
    image.image_tuning('imgonline-baseX3.jpg')
    image.borders_detection()
