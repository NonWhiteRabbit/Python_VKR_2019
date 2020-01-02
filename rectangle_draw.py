import numpy as np
import cv2
import sys, os
import matplotlib.pyplot as plt
from skimage import data, io, measure
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage.color import rgb2gray
import configparser


def image_show(image):
    """
    Функция для обработки входного изображения, прмиенения фильтра для выделения границ, а также сохранения отредактированного
    изображения в отдельный файл

    :param image: путь к файлу изображения
    :return: None
    """
    input_image = io.imread(image)
    grayscale_image = rgb2gray(input_image)
    image_threshold = grayscale_image - filters.threshold_local(grayscale_image, block_size=21, method='gaussian')
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # ax.imshow(image_threshold, cmap=plt.cm.gray)
    # plt.show()
    io.imsave('threshold_image.png', image_threshold)


def nothing(*arg):
    """
    Функция, которая необходима для работы функции image_tuning
    :param arg: *arg
    :return: None
    """
    pass


def image_tuning(image):
    """
    Функция для установки параметров HSV цветового фильтра и записи их в файл настроек

    :param image: изображение, обработанное и сохраненное в функции image_show
    :return: None
    """
    cv2.namedWindow("result")  # создаем главное окно
    cv2.namedWindow("settings")  # создаем окно настроек


    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv2.createTrackbar('h1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('h2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('v2', 'settings', 255, 255, nothing)

    while True:
        img = cv2.imread(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # считываем значения бегунков
        h1 = cv2.getTrackbarPos('h1', 'settings')
        s1 = cv2.getTrackbarPos('s1', 'settings')
        v1 = cv2.getTrackbarPos('v1', 'settings')
        h2 = cv2.getTrackbarPos('h2', 'settings')
        s2 = cv2.getTrackbarPos('s2', 'settings')
        v2 = cv2.getTrackbarPos('v2', 'settings')

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        # накладываем фильтр на кадр в модели HSV
        thresh = cv2.inRange(hsv, h_min, h_max)

        cv2.imshow('result', thresh)

        ch = cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()
    # Создаем файл настроек для сохранения в нем настроенных выше параметров HSV
    if not os.path.exists('settings.ini'):
        Config.create()
    config = Config()
    config.set_image_HSV_settings([h1, s1, v1, h2, s2, v2])
    config.change_settings()


def rectangle_draw(image):
    """
    Функция для отрисовки границ прямоугольника

    :param image: изображение, обработанное и сохраненное в функции image_show
    :return: box - координаты границ всех найденных прямоугольников
    """
    # загрузка параметров цветового фильтра из файла настроек
    config = Config()
    hsv = config.get_image_tune_settings()
    hsv_min = np.array((hsv[0], hsv[1], hsv[2]), np.uint8)
    hsv_max = np.array((hsv[3], hsv[4], hsv[5]), np.uint8)

    # чтение изображения и применение к нему настроек HSV
    img = cv2.imread(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр

    # ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # перебираем все найденные контуры в цикле
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        # отфильтровывание площадей ненужных объектов
        if 50000 < area < 60000:
            cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
            print(box)
            return box

    cv2.imshow('contours', img)  # вывод обработанного кадра в окно

    cv2.waitKey()
    cv2.destroyAllWindows()


class Config:
    def __init__(self):
        """
        Инициализация класса для создания файла конфигурации (запись значений HSV цветового фильтра)
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
        Метод для получения значений HSV цветового фильтра

        :return: list[self.image_HSV_settings]
        """
        return [self.image_h1_settings, self.image_s1_settings, self.image_v1_settings, self.image_h2_settings,
                self.image_s2_settings, self.image_v2_settings]

    def set_image_HSV_settings(self, image_hsv_settings: list):
        """
        Метод для записи значений HSV цветового фильтра

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


# rectangle_draw('logo1.png')
# image_tuning('logo1.png')