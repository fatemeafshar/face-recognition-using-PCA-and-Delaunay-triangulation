import cv2
from matplotlib import pyplot as plt
from read import read_data
from delunay2 import delaunay
from Pca import  pca
import numpy as np
# get the pathes
r = read_data()
path = "/home/fateme/Documents/archive/images/images/train/"
# path = "/home/fateme/Documents/face_recognition/105_classes_pins_dataset/"
test_number = 10
chunk = 100
normal_width = 100
normal_height = 100
num_classes = 3
pathes,classes, labels = r.read(path, chunk, test_number=test_number, num_classes=num_classes)
print(len(pathes))
print(len(classes))
delaunay_obj = delaunay()
pca_obj = pca(normal_width, normal_height)

pca_images = []
train_lables = []
landmark_sets = []
#read data
for p in range(len(pathes)):
    landmarks = delaunay_obj.get_landmarks(pathes[p])
    if landmarks != []:
        landmark_sets.append(landmarks)
        image = pca_obj.read_image(pathes[p], normal_width, normal_height)
        pca_images.append(image)
        train_lables.append(labels[p])
#train
pca_obj.train(pca_images)
delaunay_obj.train(landmark_sets)
# mean_image = pca_obj.get_mean_image(pca_images)
# # show mean image
# # plt.imshow(mean_image)
# # plt.show()
#
# U, s, VT = pca_obj.get_svd(pca_images, mean_image)
# number_eigen_values = 5
# print(U.shape)

#test
pathes,c, labels,  = r.read(path, chunk= test_number, num_classes=num_classes)
test_labels = []
test_images = []
test_landmarks = []
for i in range(len(pathes)):
    landmarks = delaunay_obj.get_landmarks(pathes[i])
    if landmarks != []:

        image = pca_obj.read_image(pathes[i], normal_width, normal_height)
        test_landmarks.append(landmarks)
        test_images.append(image)
        test_labels.append(labels[i])




wrong = 0
for i in range(len(test_images)):
    pca_test_image = pca_obj.initialize_test(test_images[i])
    delaunay_test_value = delaunay_obj.delauny_triangulation(test_landmarks[i])
    min_RV = 100000000000
    min_index = 0
    for j in range(len(pca_images)):
        eucleadian_distance = pca_obj.test(pca_test_image, j)
        distance = delaunay_obj.test(delaunay_test_value, j)
        RV = eucleadian_distance + distance/0.01
        if RV < min_RV:
            min_RV = RV
            min_index = j
    if test_labels[i] != train_lables[min_index]:
        wrong += 1
print("accuracy = ", 1- wrong/len(test_labels))


# show_images = []
# for i in range(5):
#     show_images.append(np.reshape(U[:, i], (normal_width,normal_height)))
#     # plt.imshow(np.reshape(U[:, i], (normal_width,normal_height)))
#     # plt.show()
# # U = U[:number_eigen_values, :]
# # print(U.shape)
# # for k in range(1):#len(U)):
# #     image = np.zeros((number_eigen_values**2,number_eigen_values**2))
# #     for i in range(number_eigen_values):
# #         for j in range(number_eigen_values):
# #             image[i,j] = U[k][i*number_eigen_values+j]
# #     plt.imshow(image)
# #     plt.show()
# # t = np.matmul(U.s, VT)
# # print(t)
#
# import matplotlib.pyplot as plt
# def show_images(images):#: List[np.ndarray]) -> None:
#     # n = len(images)
#     f = plt.figure()
#     for i in range(5):
#         # Debug, plot figure
#         f.add_subplot(1, 5, i + 1)
#         plt.imshow(images[i])
#
#     plt.show(block=True)
# show_images(show_images)
#
