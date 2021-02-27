import cv2
import os
from constants import GOOD_FOLDER


# Если нужно получить изображения из заранее снятого видео,
# то параметр GET_IMG_FROM_VIDEO должен быть True.
# Если нужно получить изображения из видеопотока в режиме реального времени,
# то параметр GET_IMG_FROM_VIDEO должен быть False.
GET_IMG_FROM_VIDEO = True
# Видеофайл, из которого будут извлекаться кадры
video_file = 'C:\\Users\\Natalia\\Desktop\\video_signs\\IMG_3048.mp4'


# Получение изображений из видеопотока в режиме реально времени
def get_img_real_time():
    cap = cv2.VideoCapture(1)
    # Проверка на успешное открытие камеры
    if (cap.isOpened() == False):
        print("Error opening video  file")
    # Читайте, пока видео не закончено
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        i += 1
        if ret == True:
            cv2.imwrite(GOOD_FOLDER + '\\' + str(i) + '.jpg', frame)
            # Показать полученный кадр
            cv2.imshow('Frame', frame)
            # Пока не нажата клавиша Q на клавиатуре, видео будет продолжать восроизводится
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


# Получение изображений из заранее снятого видео
def get_img_from_video_file():
    # Создание объекта VideoCapture и чтение из входного файла
    cap = cv2.VideoCapture(video_file)
    # Проверка на успешное открытие файла
    if (cap.isOpened() == False):
        print("Error opening video  file")
    # Чтение каждого кадра, пока видео не закончено
    i = 0
    while (cap.isOpened()):
        # Захват по кадрам
        ret, frame = cap.read()
        i += 1
        if ret == True:
            cv2.imwrite(os.path.join(GOOD_FOLDER, str(i) + '.jpg'), frame)
            # Показать полученный кадр
            cv2.imshow('Frame', frame)
            # Пока не нажата клавиша Q на клавиатуре, видео будет продолжать восроизводится
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    # Очистить все окна и освободить память
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # если файл существует и он не пустой
    if (os.path.exists(GOOD_FOLDER) == True) and (len(os.listdir(GOOD_FOLDER)) != 0):
        # удалить содержимое папки вместе с самой папкой
        command = 'RMDIR /s /q ' + GOOD_FOLDER
        os.system(command)
        # создать новую пустую папку
        os.mkdir(GOOD_FOLDER)
        command2 = 'cd ' + GOOD_FOLDER
        os.system(command2)
        if GET_IMG_FROM_VIDEO:
            get_img_from_video_file()
        else:
            get_img_real_time()
    elif os.path.exists(GOOD_FOLDER) == False:
        os.mkdir(GOOD_FOLDER)
        if GET_IMG_FROM_VIDEO:
            get_img_from_video_file()
        else:
            get_img_real_time()
    else:
        if GET_IMG_FROM_VIDEO:
            get_img_from_video_file()
        else:
            get_img_real_time()