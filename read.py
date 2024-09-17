import os
import pandas

import face_recognition
class read_data:
    def __init__(self):
        pass
    def read(self,path, chunk, test_number = 0, num_classes = 0):
        # path = "/home/fateme/Documents/archive/images/images/train/"
        if num_classes == 0:
            classes = os.listdir(path)
        else:
            classes = os.listdir(path)[: num_classes]
        data = []
        labels = []
        for c in range(len(classes)):
            photos = os.listdir(path+classes[c])
            for i in range(chunk):#len(photos)):
                photo = path+classes[c]+"/"+photos[i+test_number]
                data.append(photo)
                labels.append(classes[c])
        # print(data)
        # for d in data:
        #     input_image = face_recognition.load_image_file(d)
        #     face_landmarks = face_recognition.face_landmarks(input_image)
        #     print(len(face_landmarks))
        return data, classes, labels

# r = read_data()
#
# r.read(6)

# path = "/home/fateme/Documents/archive/images/images/train/"
# files = [f for f in os.listdir(path) if os.path.isfile(f)]
# print(files)














