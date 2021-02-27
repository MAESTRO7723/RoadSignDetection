import os
import cv2
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def crop_frame(image):
    if (len(image.shape) == 3):
        height, length, _ = image.shape
    else:
        height, length = image.shape
    return image[0: height // 3 * 2, 0: length]


if __name__ == '__main__':
    # путь к видеозаписи
    filepath = 'C:\\Users\\Natalia\\Desktop\\video_signs\\IMG_3050.mp4'
    # загрузка обученных классификаторов
    sign1_cascade = cv2.CascadeClassifier(
        'D:\\PythonProjects\\DETECT_TRAFFIC_SIGN\\Haar_cascad\\signs\\kirpich\\1\\data\\kirpich.xml')
    # чтение видеозаписи
    cap = cv2.VideoCapture(filepath)
    # либо чтение видеопотока в реальном времени
    # cap = cv2.VideoCapture(1)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if (ret):
            # обрезаем каждый кадр
            cropped_frame = crop_frame(frame)
            # переводим кадр в ч/б
            gray_filered = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            # функция поиска знаков
            signs_kirpich = sign1_cascade.detectMultiScale(gray_filered, scaleFactor=1.15, minNeighbors=12,
                                                           minSize=(15, 15))
            # рисование прямоугольника вокруг найденного объекта
            for (x, y, w, h) in signs_kirpich:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.putText(frame, "NO ENTRY", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            out.write(frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        else:
            break
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

