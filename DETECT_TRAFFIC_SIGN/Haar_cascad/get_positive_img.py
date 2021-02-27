import cv2
import os
from constants import GOOD_FOLDER

# список опорных точек
refPt = []
# указание выполняется ли обрезка или нет
cropping = False


def click_and_crop(event, x, y):
    # глобальные переменные
    global refPt, cropping
    # если была нажата левая кнопка мыши, то начальные
    # координаты (x, y) записываются и указывается, что выполняется обрезка
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # проверка на то, что левая кнопка мыши была отпущена
    elif event == cv2.EVENT_LBUTTONUP:
        # запись конечных координат (x, y) и указание того, что обрезка завершена
        refPt.append((x, y))
        cropping = False
        # рисование прямоугольника вокруг выделенной области
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)


def crop(image, file):
    # цикл продолжается пока не нажата клавиша
    while True:
        # вывод изображения на экран и ожидание нажатия клавиши
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # если нажата клавиша "r", то выделенная область обрезки сбрасывается
        if key == ord("r"):
            image = clone.copy()
        # при нажатии клавиши "c" происходит выход из цикла
        elif key == ord("c"):
            break
    # если есть 2 опорные точки, то выделенная область обрезается
    # и выводится на экран
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imwrite(os.path.join(GOOD_FOLDER, file), roi)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    for root, dirs, files in os.walk(GOOD_FOLDER):
        for file in files:
            # чтение каждого изображения в указанной папке
            file_path = os.path.join(GOOD_FOLDER, file)
            image = cv2.imread(file_path)
            # изменение размера изображения для того, чтобы оно убралось на весь экран
            final_wide = 300
            r = float(final_wide) / image.shape[1]
            dim = (final_wide, int(image.shape[0] * r))
            image = cv2.resize(image, dim)
            clone = image.copy()
            cv2.namedWindow("image")
            # функция вызова мыши
            cv2.setMouseCallback("image", click_and_crop)
            crop(image, file)
