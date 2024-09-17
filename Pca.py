import math

import dlib
import cv2
import face_recognition
import numpy as np
from scipy.spatial import Delaunay
from matplotlib import image
from matplotlib import pyplot as plt

from numpy import array
from scipy.linalg import svd


class pca:
    def __init__(self, normal_width, normal_height):
        self.normal_width = normal_width
        self.normal_height = normal_height

    def previous_train(self, pathes):
        img = cv2.imread(pathes[0], cv2.IMREAD_GRAYSCALE)
        row = len(img)
        col = len(img[0])
        number_of_images = len(pathes)
        train_images = np.zeros((number_of_images, row * col))
        self.mean = np.zeros((1, row*col))
        for p in range(number_of_images):
            img = cv2.imread(pathes[p], cv2.IMREAD_GRAYSCALE)
            for i in range(0, row):
                for j in range(0, col):
                    train_images[p, i*col + j] = img[i, j]
                    self.mean[p, i*col + j] += img[i, j]
        i_images = np.zeros((number_of_images, row * col))
        for i in range(number_of_images):
            for j in range(row*col):
                i_images[i, j] = self.mean[i,j] - train_images
        i_images_transpose = i_images.transpose()
        covariance = np.matmul(i_images, i_images_transpose)
        U, s, VT = svd(array(covariance))
        print(U)
        print(s)
        print(VT)

        # find best eigen value

    def read_image(self, path, width, height):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (width, height))
        return image

    def get_mean_image(self, images):#, width, height):
        self.number_train_images = len(images)
        mean = np.zeros((len(images[0]), len(images[0][0])))
        for img in images:

            for i in range(0, len(img)):
                for j in range(0, len(img[0])):
                    # print(img[i, j] )
                    mean[i, j] += img[i, j]

        for i in range(len(mean)):
            for j in range(len(mean[0])):
                mean[i, j] = int(mean[i, j] / len(images))
        return mean
    def compare_images(self, raw_images, U_images, eigenvalue_images, U_k):
        fig = plt.figure(figsize=(10, 7))

        # setting values to rows and column variables
        rows = 5
        columns = 2
        for i in range(rows):
            fig.add_subplot(rows, columns, columns*i+1)
            # showing image
            image1 = raw_images[i]
            plt.imshow(image1)
            plt.axis('off')
            plt.title("initial image")

            # fig.add_subplot(rows, columns, columns*i+2)
            # # showing image
            # image2 = np.reshape(U_images[:, i], (self.normal_width, self.normal_height))
            # plt.imshow(image2)
            # plt.axis('off')
            # plt.title("U matrix ")

            fig.add_subplot(rows, columns, columns*i+2)
            # showing image
            print("shapes",U_k.shape, eigenvalue_images[:, i].shape)
            approximated_face =np.matmul(U_k, eigenvalue_images[:, i])
            image3 = np.reshape(approximated_face, (self.normal_width, self.normal_height))
            plt.imshow(image3)
            plt.axis('off')
            plt.title("eigenface")
        plt.show()

    def get_svd(self, images, mean):
        # I_matix_set = []  # np.zeros((48, 48))
        # images_matix = np.zeros((len(images), len(mean) * len(mean[0])))
        images_matix = np.zeros((len(mean) * len(mean[0]), len(images)))
        for k in range(len(images)):
            centeralized_image = mean - images[k]
            hh = np.reshape(centeralized_image, (1, len(mean) * len(mean[0])))
            images_matix[:, k] = hh
            # for i in range(len(mean)):
            #     for j in range(len(mean[0])):
            #         # images_matix[k, i*len(mean[0])+j] = mean[i, j] - images[k][i][ j]
            #         # image = np.reshape(images[k], (1, i*j))
            #         images_matix[ i * len(mean[0]) + j, k] = mean[i, j] - images[k][i][j]
            # I_matix_set.append(I_matix)
        # print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
        # for im in images_matix:
        #     print(im)
        # print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")


        U, s, VT = svd(array(images_matix))
        print(U)
        print(s)
        print(VT)

        print(mean)
        return U, s, VT

    def train(self, images_set):
        self.mean_image = self.get_mean_image(images_set)
        self.U, s, VT = self.get_svd(images_set, self.mean_image)
        k_best_eigenvalues = 50
        # Utranspose = self.U.transpose()
        # self.Utranspose_k = Utranspose[:k_best_eigenvalues, :]
        U_k = self.U[:, :k_best_eigenvalues]
        self.Utranspose_k = U_k.transpose()
        self.eigenfaces = np.zeros((k_best_eigenvalues , self.number_train_images))
        for i in range(self.number_train_images):
            face =  np.reshape(images_set[i], (self.normal_width*self.normal_height, ))#self.U[:, i]
            # np.dot(face[0, :], np.ones(i))
            self.eigenfaces[:, i] = np.matmul(self.Utranspose_k, face)
        print("done", self.Utranspose_k.shape, U_k.shape, self.eigenfaces.shape)
        self.compare_images(images_set, self.U, self.eigenfaces, U_k)
    def initialize_test(self, test_image):
        test_image = self.mean_image - test_image
        x = np.reshape(test_image, ( self.normal_width * self.normal_height,1))
        x = np.matmul(self.Utranspose_k, x)
        return x
    def test(self, test_image, train_image_index):

        euclidain_distance = 0
        for j in range(len(test_image)):
            euclidain_distance += (self.eigenfaces[j][train_image_index] - test_image[j]) ** 2
        return euclidain_distance


    # delaunay_tirangulation(get_landmarks(path))
    # pca([path, path])
#
# path = "/home/fateme/Documents/face_recognition/105_classes_pins_dataset/pins_Adriana Lima/Adriana Lima31_154.jpg"
#
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# print(len(img), len(img[0]))




# import cv2 as cv
# import glob
# import matplotlib.pyplot as plt
#
# path = "/home/fateme/Documents/face_recognition/105_classes_pins_dataset/pins_Adriana Lima/Adriana Lima232_133.jpg"
# pca_object = pca()
# # image = pca_object.read_image(path, 48, 48)
# my_image_list = [pca_object.read_image(path, 48, 48)]#, pca_object.get_mean_image([image], 48, 48)]
#
# # for file in glob.glob(path):
# #     file = cv.imread(file) # BGR
# #     # convert BGR to RGB
# #     rgb_image = cv.cvtColor(file, cv.COLOR_BGR2RGB)
# #     my_image_list.append(rgb_image)
#
# # display all images
# plt.figure(figsize=(20,10))
#
# columns = 1
#
# for i, image in enumerate(my_image_list):
#     plt.subplot(len(my_image_list) / columns + 1, columns, i + 1)
#     plt.imshow(ipca_object = pca()
# image = pca_object.read_image(path, 48, 48)mage)



import cv2


# # Load an image
# # img = cv2.imread('/home/fateme/Documents/face_recognition/105_classes_pins_dataset/pins_Adriana Lima/Adriana Lima232_133.jpg')
# path = '/home/fateme/Documents/face_recognition/105_classes_pins_dataset/pins_Adriana Lima/Adriana Lima232_133.jpg'
# from matplotlib import image
# from matplotlib import pyplot as plt
#
#
# pca_object = pca()
# img = pca_object.read_image(path, 224, 224)
# pca_object.get_mean_image([img, img/1.3, img/2])
# plt.imshow(img)
# plt.show()
