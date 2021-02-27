import cv2
from constants_lines import *
import numpy as np


def draw_lines(img, lines, color=(255, 0, 0), thickness=10):
    # изменяем размер массива линий, делая его двухмерным
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    # lines - массив линий, линия состоит из двух точек (x1,y1) и (x2,y2)
    # меняем массив линий на массив отдельных точек
    lines = np.reshape(lines, (lines.shape[0] * 2, 2))
    # считаем, что если точка находится справа от середины изображения,
    # то она принадлежит к правой полосе
    right_lines = np.array(list(filter(lambda x: x[0] > (img.shape[1] / 2), lines)))
    # считаем, что если точка находится слева от середины изображения,
    # то она принадлежит к левой полосе
    left_lines = np.array(list(filter(lambda x: x[0] < (img.shape[1] / 2), lines)))
    if len(right_lines) != 0 and len(left_lines) != 0 and len(lines) != 0:
        # находим нижнюю точку правой полосы
        min_right_x, min_right_y = np.amin(right_lines, axis=0)
        # находим нижнюю точку левой полосы
        min_left_x, min_left_y = np.amin(left_lines, axis=0)

        # находим аппроксимирующую функцию для правой и левой полосы по методу наименьших квадратов
        right_curve = np.poly1d(np.polyfit(right_lines[:, 1], right_lines[:, 0], 2))
        left_curve = np.poly1d(np.polyfit(left_lines[:, 1], left_lines[:, 0], 2))

        # Решаем полином, вычисляем максимальное и минимальное значения по x
        max_right_x = int(right_curve(img.shape[0]))
        min_right_x = int(right_curve(min_right_y))

        # Решаем полином, вычисляем значения x
        min_left_x = int(left_curve(img.shape[0]))
        max_left_x = int(left_curve(min_left_y))

        # находим самую низкую точку, для того чтобы полосы
        # начинались на одном уровне. А значение самой высокой точки
        # будет равно значению высоты изображения, для того чтобы линии
        # были отображены до конца изображения
        min_y = min(min_right_y, min_left_y)

        r1 = (min_right_x, min_y)
        r2 = (max_right_x, img.shape[0])
        cv2.line(img, r1, r2, color, thickness)

        l1 = (max_left_x, min_y)
        l2 = (min_left_x, img.shape[0])
        cv2.line(img, l1, l2, color, thickness)

        print(r1, '   ', r2)
        print(l1, '   ', l2)


def search_lines(img):
    # преобразование Хафа для поиска прямых линий (возвращает 2 конечные точки линий)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    # создаем массив такого же размера как исходное изображение, заполненный нулями
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return line_img, lines


def region_of_interest(img):
    # размер изображения, h-высота, w-ширина
    h = img.shape[0]
    w = img.shape[1]
    # определение вершин для области интереса
    # (левая верхняя точка исходного изображения имеет координаты (0;0),
    # правая нижняя точка исходного изображения имеет координаты (x;y)).
    print(w,h)
    # левая нижняя точка области интереса
    bottom_left = (200, h)
    # левая верхняя точка области интереса
    top_left = (300, 50)
    # правая верхняя точка области интереса
    top_right = (600, 50)
    # правая нижняя точка области интереса
    bottom_right = (700, h)

    # создание массива, определяющего область интереса на изображении
    vertices = np.array([[bottom_left,
                          top_left,
                          top_right,
                          bottom_right]],
                        dtype=np.int32)
    # изображение для рисования (маска)
    image_to_draw = np.zeros_like(img)
    # определения цвета для закрашивания области интереса
    # в зависимости от входного изображения (одноканальное/многоканальное)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # количество каналов у изображения (3 или 4)
        color = (255,) * channel_count
    else:
        color = 255
    # закрашивание полигона, определенного вершинами цветом заливки
    cv2.fillPoly(image_to_draw, vertices, color)
    # наложение маски поверх оригинального изображения, сохраняя каждый пиксель
    # изображения, если соответствующее значение маски равно 1
    image_mask = cv2.bitwise_and(img, image_to_draw)
    return image_mask


def prepare_img(img):
    # перевод изображения в черно-белое пространство
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # применение к изображению фильтра Гаусса
    blur_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    # применение к изображению детектора границ Кэнни
    Canny_img = cv2.Canny(np.uint8(blur_img), low_threshold, high_threshold)
    return Canny_img


if __name__ == "__main__":
    # путь к изображению
    path_img = 'D:\\PythonProjects\\detect_lines\\Lane-Detection-master\\Lane-Detection-master\\test_images\\road6.jpg'
    # чтение изображения
    image = cv2.imread(path_img)
    # проверка существует ли изображение
    if image is None:
        print('Изображение отсутствует. Проверьте путь к изображению.')
    final_wide = 960
    r = float(final_wide) / image.shape[1]
    dim = (final_wide, int(image.shape[0] * r))

    # уменьшаем изображение до подготовленных размеров
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # предобработка изображения
    Canny_img = prepare_img(resized)
    # выделение области интереса
    img_mask = region_of_interest(Canny_img)
    cv2.imshow('can', img_mask)
    cv2.waitKey(0)
    # поиск линий на изображении
    line_img, lines = search_lines(img_mask)
    # рисование линий на изображении
    draw_lines(line_img, lines, thickness=10)
    lines_edges = cv2.addWeighted(resized, 0.8, line_img, 1, 0)
    cv2.imshow('lines', lines_edges)
    cv2.waitKey(0)
    cv2.imwrite('lines_output.jpg', lines_edges)

