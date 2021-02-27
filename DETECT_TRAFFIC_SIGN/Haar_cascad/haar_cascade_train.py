import cv2
import os
import subprocess


# для обучения каскада Хаара нужно раскоментировать одну из трех строчек
# например, если мы хотим обучить каскад для дорожной разметки, то нужно
# раскоментировать строку from constants_lines import *
from constants import *     # константы


# изменяем размер отрицательных изображений
def resize_bad_images():
    for img in os.listdir(BAD_FOLDER):
        _, file_extension = os.path.splitext(img)
        file_extension = file_extension.lower()
        if file_extension.endswith("jpg") or file_extension.endswith("png"):
            try:
                img_read = cv2.imread("%s/%s" % (BAD_FOLDER, img))
                resized_image = cv2.resize(img_read, (100, 100))
                cv2.imwrite("%s/%s" % (BAD_FOLDER, img), resized_image)

            except Exception as e:
                print(str(e))

def create_file_if_its_nul(file):
    if os.path.isfile(file) == 1 and os.stat(file).st_size != 0:
        command = 'TYPE nul > ' + file
        # очистить содержимое файла
        os.system(command)


# создание текстового файла с описанием отрцательных изображений
def create_bad_txt():
    for root, dirs, files in os.walk(BAD_FOLDER):
        for file in files:
            file_path = os.path.join(BAD_FOLDER, file)
            # создать файл с описанием отрцательных изображений
            my_file = open(BAD_TXT_FILE, "a")
            my_file.write(file_path + '\n')
            my_file.close()


# создание текстового файла с описанием положительных изображений
def create_good_txt():
    for root, dirs, files in os.walk(GOOD_FOLDER):
        for file in files:
            file_path = os.path.join(GOOD_FOLDER, file)
            img = cv2.imread(file_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(GOOD_FOLDER, file), img_gray)
            image_shape = img.shape
            # создать файл с описанием отрцательных изображений
            my_file = open(GOOD_TXT_FILE, "a")
            my_file.write(file_path + ' 1 0 0 ' + str(image_shape[1]) + ' ' + str(image_shape[0]) + '\n')
            my_file.close()


# создание пачки приведённых положительных изображений
def get_vector_file():
    subprocess.check_call(["D:\\opencv\\build\\x64\\vc12\\bin\\opencv_createsamples",
                           "-info", GOOD_TXT_FILE,
                           "-vec", VEC_FILE,
                           "-w", str(GOOD_SIZE[0]),
                           "-h", str(GOOD_SIZE[1])])


# создаем итоговый каскад
def train_haar_cascade():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    subprocess.check_call(["D:\\opencv\\build\\x64\\vc12\\bin\\opencv_traincascade",
                           "-data", DATA_FOLDER,
                           "-vec", VEC_FILE,
                           "-bg", BAD_TXT_FILE,
                           "-numStages", str(NUM_STAGES),
                           "-minHitRate", str(MIN_HIT_RATE),
                           "-maxFalseAlarmRate", str(MAX_FALSE_ALARM_RATE),
                           "-numPos", str(NUMGOOD),
                           "-numNeg", str(NUMBAD),
                           "-w", str(GOOD_SIZE[0]),
                           "-h", str(GOOD_SIZE[1]),
                           "-mode", MODE,
                           "-precalcValBufSize", str(MEMORY_SIZE),
                           "-precalcIdxBufSize", str(MEMORY_SIZE)])


if __name__ == '__main__':
    resize_bad_images()
    create_file_if_its_nul(BAD_TXT_FILE)
    create_bad_txt()
    create_file_if_its_nul(GOOD_TXT_FILE)
    create_good_txt()
    create_file_if_its_nul(VEC_FILE)
    get_vector_file()
    train_haar_cascade()
