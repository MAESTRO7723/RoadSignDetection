import cv2
from detect_line import draw_lines, search_lines, region_of_interest, prepare_img


video_file = 'C:\\Users\\Natalia\\Desktop\\video_signs\\IMG_3163.mp4'
# Создание объекта VideoCapture и чтение из входного файла

cap = cv2.VideoCapture(video_file)
#cap = cv2.VideoCapture(1)
# Проверка на успешное открытие файла
if (cap.isOpened() == False):
    print("Error opening video  file")
# Чтение каждого кадра, пока видео не закончено
i = 0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
while (cap.isOpened()):
    # Захват по кадрам
    ret, frame = cap.read()
    i += 1
    if ret == True:
        final_wide = 960
        r = float(final_wide) / frame_width
        dim = (final_wide, int(frame_height * r))

        # уменьшаем изображение до подготовленных размеров
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # предобработка изображения
        Canny_img = prepare_img(resized)
        # выделение области интереса
        img_mask = region_of_interest(Canny_img)
        # поиск линий на изображении
        line_img, lines = search_lines(img_mask)
        if lines is not None:
            # рисование линий на изображении
            draw_lines(line_img, lines, thickness=15)
            lines_edges = cv2.addWeighted(resized, 0.8, line_img, 1, 0)
            # Показать полученный кадр
            out.write(lines_edges)
            cv2.imshow('Frame', lines_edges)
        else:
            cv2.imshow('Frame', resized)

        # Пока не нажата клавиша Q на клавиатуре, видео будет продолжать восроизводится
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
# Очистить все окна и освободить память
cap.release()
cv2.destroyAllWindows()
