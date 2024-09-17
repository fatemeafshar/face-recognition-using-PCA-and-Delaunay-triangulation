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
path = "/home/fateme/Documents/archive/images/images/train/angry/186.jpg"


class delaunay:
    def __int__(self):
        pass

    def get_landmarks(self, path):
        try:
            input_image = face_recognition.load_image_file(path)
        except:
            print("An exception occurred")
            return []
        try:
            face_landmarks = face_recognition.face_landmarks(input_image)
        except:
            print("An exception occurred")
            return []
        landmark_points = []
        if face_landmarks == {} or face_landmarks == []:
            # print("kk")
            return []
        for key in face_landmarks[0]:
            # print(face_landmarks[0][key])
            for point in face_landmarks[0][key]:
                landmark_points.append(point)
        return landmark_points



    def train_triangulation(self, pathes):
        self.train_averages = []
        for path in pathes:
            self.train_averages.append(self.delaunay_tirangulation(path))
        return self.train_averages
    def delauny_triangulation(self, landmark_points):
        tri = Delaunay(landmark_points)
        a_max = 0
        a_set = []
        for triangle in tri.simplices:
            p = landmark_points[triangle[0]]
            q = landmark_points[triangle[1]]
            r = landmark_points[triangle[2]]
            l1 = math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
            l2 = math.sqrt((p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2)
            l3 = math.sqrt((r[0] - q[0]) ** 2 + (r[1] - q[1]) ** 2)
            s = (l1 + l2 + l3) / 2
            a = math.sqrt(s * (s - l1) * (s - l2) * (s - l3))
            if a > a_max:
                a_max = a
            a_set.append(a)
        ra_set = []
        ra_avg = 0
        for a in a_set:
            ra = a / a_max
            ra_set.append(ra)
            ra_avg += ra
        ra_avg = ra_avg / len(a_set)
        return ra_avg
    def train(self,landmark_sets):
        ra_avg_set = []
        for landmark_points in landmark_sets:
            ra_avg = self.delauny_triangulation(landmark_points)
            ra_avg_set.append(ra_avg)
        self.ra_avg_set = ra_avg_set
        # return ra_avg_set

    def test(self, test_value, train_value_index):
        return (test_value - self.ra_avg_set[train_value_index])**2
