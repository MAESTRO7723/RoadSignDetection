import cv2


def crop_frame(image):
    if (len(image.shape) == 3):
        height, length, _ = image.shape
    else:
        height, length = image.shape
    return image[0: height // 3 * 2, 0: length]


if __name__ == '__main__':
    # путь к видеозаписи
    filepath = 'C:\\Users\\Natalia\\Desktop\\video_signs\\IMG_3146.mp4'
    # загрузка обученных классификаторов
    sign1_cascade = cv2.CascadeClassifier('D:\\PythonProjects\\DETECT_TRAFFIC_SIGN\\Haar_cascad\\signs\\kirpich\\1\\data\\kirpich.xml')
    sign2_cascade = cv2.CascadeClassifier('D:\\PythonProjects\\DETECT_TRAFFIC_SIGN\\Haar_cascad\\signs\\nalevo\\1\\data\\nalevo.xml')
    sign3_cascade = cv2.CascadeClassifier('D:\\PythonProjects\\DETECT_TRAFFIC_SIGN\\Haar_cascad\\signs\\napravo\\1\\data\\napravo.xml')
    sign4_cascade = cv2.CascadeClassifier('D:\\PythonProjects\\DETECT_TRAFFIC_SIGN\\Haar_cascad\\signs\\pryamo\\1\\data\\pryamo.xml')
    sign5_cascade = cv2.CascadeClassifier('D:\\PythonProjects\\DETECT_TRAFFIC_SIGN\\Haar_cascad\\signs\\pryamonalevo\\1\\data\\pryamonalevo.xml')
    sign6_cascade = cv2.CascadeClassifier('D:\\PythonProjects\\DETECT_TRAFFIC_SIGN\\Haar_cascad\\signs\\pryamonapravo\\1\\data\\pryamonapravo.xml')
    light_green = cv2.CascadeClassifier('D:\\PythonProjects\\DETECT_TRAFFIC_SIGN\\Haar_cascad\\light\\green\\1\\data\\green.xml')
    light_red = cv2.CascadeClassifier('D:\\PythonProjects\\DETECT_TRAFFIC_SIGN\\Haar_cascad\\light\\red\\1\\data\\red.xml')
    # чтение видеозаписи
    cap = cv2.VideoCapture(filepath)
    # либо чтение видеопотока в реальном времени
    #cap = cv2.VideoCapture(1)
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
            #scaler = StandardScaler()
            #gray = scaler.fit_transform(gray_filered)
            # функция поиска знаков
            signs_kirpich = sign1_cascade.detectMultiScale(gray_filered, scaleFactor=1.15, minNeighbors=15,
                                                            minSize=(200, 200))
            signs_nalevo = sign2_cascade.detectMultiScale(gray_filered, scaleFactor=1.15, minNeighbors=20,
                                                        minSize=(200, 200))
            signs_napravo = sign3_cascade.detectMultiScale(gray_filered, scaleFactor=1.15, minNeighbors=20,
                                                        minSize=(200, 200))
            signs_pryamo = sign4_cascade.detectMultiScale(gray_filered, scaleFactor=1.15, minNeighbors=20,
                                                        minSize=(200, 200))
            signs_pryamonalevo = sign5_cascade.detectMultiScale(gray_filered, scaleFactor=1.15, minNeighbors=20,
                                                        minSize=(200, 200))
            signs_pryamonapravo = sign6_cascade.detectMultiScale(gray_filered, scaleFactor=1.15, minNeighbors=20,
                                                        minSize=(200, 200))
            # функция поиска светофоров
            light_greeen = light_green.detectMultiScale(gray_filered, scaleFactor=1.15, minNeighbors=15,
                                                        minSize=(300, 600))
            light_reed = light_red.detectMultiScale(gray_filered, scaleFactor=1.15, minNeighbors=15,
                                                        minSize=(300, 600))
            # рисование прямоугольника вокруг найденного объекта
            for (x, y, w, h) in signs_kirpich:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.putText(frame, "NO ENTRY", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
            for (x, y, w, h) in signs_nalevo:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(frame, "LEFT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            for (x, y, w, h) in signs_napravo:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, "RIGHT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            for (x, y, w, h) in signs_pryamo:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
                cv2.putText(frame, "STRAIGHT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
            for (x, y, w, h) in signs_pryamonalevo:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
                cv2.putText(frame, "STRAIGHT AND LEFT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
            for (x, y, w, h) in signs_pryamonapravo:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(frame, "STRAIGHT AND RIGHT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
            for (x, y, w, h) in light_greeen:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.putText(frame, "GREEN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
            for (x, y, w, h) in light_reed:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.putText(frame, "RED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
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

