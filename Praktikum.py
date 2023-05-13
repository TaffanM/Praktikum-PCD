import math
import sys
import dlib
import cv2
import imutils
import numpy
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
from skimage import data, exposure
from skimage.feature import hog
from time import sleep

import konvolusi

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('test.ui', self)
        self.image = None
        # Load Gambar
        self.button_load.clicked.connect(self.loadClicked)
        self.saveImage.clicked.connect(self.saveClicked)

        # Operasi Titik
        self.grayButton.clicked.connect(self.grayimage)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionOperasi_Kontras.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionNegative_Image.triggered.connect(self.negativeImage)
        self.actionBiner_Image.triggered.connect(self.binerImage)

        # Operasi Histogram
        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogram)

        # Operasi Geometri
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_Derajat.triggered.connect(self.rotasi90)
        self.action_90_Derajat.triggered.connect(self.rotasiMin90)
        self.action45_Derajat.triggered.connect(self.rotasi45)
        self.action_45_Derajat.triggered.connect(self.rotasiMin45)
        self.action180_Derajat.triggered.connect(self.rotasi180)
        self.action2x.triggered.connect(self.zoomIn2x)
        self.action3x.triggered.connect(self.zoomIn3x)
        self.action4x.triggered.connect(self.zoomIn4x)
        self.action1_4x.triggered.connect(self.zoomOut14x)
        self.action1_2x.triggered.connect(self.zoomOut12x)
        self.action3_4x.triggered.connect(self.zoomOut34x)
        self.actionCrop.triggered.connect(self.crop)

        # Operasi Aritmatika
        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika)
        self.actionKali_dan_Bagi.triggered.connect(self.aritmatikaKali)
        self.actionOperasi_AND.triggered.connect(self.operasiAND)
        self.actionOperasi_OR.triggered.connect(self.operasiOR)
        self.actionOperasi_XOR.triggered.connect(self.operasiXOR)

        # Operasi Spasial
        self.actionKonvolusi_2D.triggered.connect(self.konvolusi2D)
        self.actionMean_Filter.triggered.connect(self.meanFilter)
        self.actionGaussian_Filter.triggered.connect(self.gaussianFilter)
        self.actionImage_Sharpening.triggered.connect(self.imageSharpening)
        self.actionMedian_Filter.triggered.connect(self.medianFilter)
        self.actionMax_Filter.triggered.connect(self.maxFilter)

        # Transformasi
        self.actionSmooth.triggered.connect(self.transformasiSmooth)
        self.actionEdge.triggered.connect(self.tranformasiEdge)

        # Deteksi Tepi
        self.actionSobel.triggered.connect(self.deteksiSobel)
        self.actionPrewit.triggered.connect(self.deteksiPrewit)
        # self.actionRobets.triggered.connect(self.deteksiRobets)
        self.actionCanny_Edge.triggered.connect(self.cannyEdge)

        # Morfologi
        self.actionMorfologi.triggered.connect(self.morfologi)

        # Thresholding
        self.actionBinary_2.triggered.connect(self.thresholdBinary)
        self.actionBinary_Invers.triggered.connect(self.thresholdBinaryInvers)
        self.actionTrunc.triggered.connect(self.thresholdTrunc)
        self.actionTo_Zero.triggered.connect(self.thresholdToZero)
        self.actionTo_Zero_Invers.triggered.connect(self.thresholdToZeroInvers)

        # Adaptive Thresholding
        self.actionMean_Thresholding.triggered.connect(self.thresholdMean)
        self.actionGaussian_Thresholding.triggered.connect(self.thresholdGaussian)
        self.actionOtsu_Thresholding.triggered.connect(self.thresholdOtsu)
        self.actionContour.triggered.connect(self.contour)

        # Color Processing
        self.actionColor_Tracking.triggered.connect(self.colorTracking)
        self.actionColor_Picker.triggered.connect(self.colorPicker)

        # Haar Cascade
        self.actionObject_Detection.triggered.connect(self.objectDetection)
        self.actionHistogram_of_Gradient.triggered.connect(self.HOG)
        self.actionHaar_Cascade_Face_Eye_Detection.triggered.connect(self.HaarFaceEye)
        self.actionHaar_Cascade_Pedestrian_Detection.triggered.connect(self.HaarPedestrian)
        self.actionCircle_Hough_Transform.triggered.connect(self.circleHough)
        self.actionHistogram_of_Gradient_Pedestrian.triggered.connect(self.HOGPedestrian)

        # Face Detection
        self.actionFacial_Landmark.triggered.connect(self.facialLandmark)
        self.actionSwap_Face.triggered.connect(self.swapFace)
        self.actionSwap_Face_Real_Time.triggered.connect(self.swapFaceRealTime)
        self.actionYawn_Detection.triggered.connect(self.yawnDetection)


    # Pertemuan 2

    #
    # Praktek A2
    #
    def loadClicked(self):
        self.image = cv2.imread('img/kidney-diagram.png')
        self.displayImage(1)

    def saveClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG (*.jpg *.jpeg);;PNG (*.png)",
                                                   options=options)

        if file_name:
            self.image.save(file_name)

    #
    # Praktek A3
    #
    def grayimage(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.image[i, j, 0] +
                    0.587 * self.image[i, j, 1] +
                    0.114 * self.image[i, j, 2], 0, 255)

        self.image = gray
        self.displayImage(2)

    #
    # Praktek A4
    #
    def brightness(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        brightness = 80
        for i in np.arange(H):
            for j in np.arange(W):
                a = self.image.item(i, j)
                b = a + brightness
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0
                else:
                    b = b
                self.image.itemset((i, j), b)

        self.displayImage(2)

    #
    # Praktek A5
    #
    def contrast(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.image.itemset((i, j), b)

        self.displayImage(2)

    #
    # Praktek A6
    #
    def contrastStretching(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        minV = np.min(self.image)
        maxV = np.max(self.image)
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.image.itemset((i, j), b)

        self.displayImage(2)

    #
    # Praktek A7
    #
    def negativeImage(self):
        maximum_intensity = 255

        H, W = self.image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = math.ceil(255 - self.image.item(i, j))

                self.image.itemset((i, j), b)

        self.displayImage(2)

    #
    # Praktek A8
    #
    def binerImage(self):
        p = 128
        H, W = self.image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a, 0, 255)
                if b > p:
                    a = 255
                elif b < p:
                    b = 1
                else:
                    b = 0
                self.image.itemset((i, j), b)
        self.displayImage(2)

    # Pertemuan 3

    #
    # Praktek A9
    #
    def grayHistogram(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.image[i, j, 0] +
                    0.587 * self.image[i, j, 1] +
                    0.114 * self.image[i, j, 2], 0, 255)

        self.image = gray
        self.displayImage(2)
        plt.hist(self.image.ravel(), 255, [0, 255])
        plt.show()

    #
    # Praktek A10
    #
    def RGBHistogram(self):
        color = ("b", "g", "r")
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.image], [i], None, [256], [0, 256])
            plt.plot(histo, color=col)
            plt.xlim([0, 256])
        self.displayImage(2)
        plt.show()

    #
    # Praktek A11
    #
    def EqualHistogram(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")
        self.image = cdf[self.image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color="b")
        plt.hist(self.image.flatten(), 256, [0, 256], color="r")
        plt.xlim([0, 256])
        plt.legend(("cdf", "histogram"), loc="upper left")
        plt.show()

    #
    # Praktek B1
    #
    def translasi(self):
        H, W = self.image.shape[:2]
        quarter_h, quarter_w = H / 4, W / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.image, T, (W, H))
        self.image = img
        self.displayImage(2)

    #
    # Praktek B2
    #
    def rotasi90(self):
        self.rotasi(90)

    def rotasiMin90(self):
        self.rotasi(-90)

    def rotasi45(self):
        self.rotasi(45)

    def rotasiMin45(self):
        self.rotasi(45)

    def rotasi180(self):
        self.rotasi(180)

    def rotasi(self, degree):
        H, W = self.image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((W / 2, H / 2), degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((H * sin) + (W * cos))
        nH = int((H * cos) + (W * sin))

        rotationMatrix[0, 2] += (nW / 2) - W / 2
        rotationMatrix[1, 2] += (nH / 2) - H / 2

        rot_image = cv2.warpAffine(self.image, rotationMatrix, (H, W))
        self.image = rot_image
        self.displayImage(2)

    #
    # Praktek B3
    #
    def zoomIn2x(self):
        self.zoomIn(2)

    def zoomIn3x(self):
        self.zoomIn(3)

    def zoomIn4x(self):
        self.zoomIn(4)

    def zoomIn(self, skala):
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def zoomOut12x(self):
        self.zoomOut(0.5)

    def zoomOut14x(self):
        self.zoomOut(0.25)

    def zoomOut34x(self):
        self.zoomIn(0.75)

    def zoomOut(self, skala1):
        resize_image = cv2.resize(self.image, None, fx=skala1, fy=skala1, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    #
    # Praktek B4
    #
    def crop(self):
        cropped_image = self.image[0:500, 0:300]
        cv2.imshow('Original', self.image)
        cv2.imshow('Crop', cropped_image)
        cv2.waitKey(0)

    #
    # Praktek C2
    #
    def operasiAND(self):
        image1 = cv2.imread('img/01.png', 1)
        image2 = cv2.imread('img/1.jpg', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        and_op = cv2.bitwise_and(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi AND', and_op)
        cv2.waitKey(0)

    def operasiOR(self):
        image1 = cv2.imread('img/01.png', 1)
        image2 = cv2.imread('img/1.jpg', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        or_op = cv2.bitwise_or(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi OR', or_op)
        cv2.waitKey(0)

    def operasiXOR(self):
        image1 = cv2.imread('img/01.png', 1)
        image2 = cv2.imread('img/1.jpg', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        xor_op = cv2.bitwise_xor(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi OR', xor_op)
        cv2.waitKey(0)

    #
    # Praktek C1
    #
    def aritmatika(self):
        image1 = cv2.imread('img/01.png', 0)
        image2 = cv2.imread('img/1.jpg', 0)
        image_plus = image1 + image2
        image_minus = image1 - image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Tambah', image_plus)
        cv2.imshow('Image Kurang', image_minus)
        cv2.waitKey(0)

    def aritmatikaKali(self):
        image1 = cv2.imread('img/01.png', 0)
        image2 = cv2.imread('img/1.jpg', 0)
        image_mul = image1 * image2
        image_div = image1 / image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Kali', image_mul)
        cv2.imshow('Image Bagi', image_div)
        cv2.waitKey(0)

    # Pertemuan 4

    #
    # Praktek D1
    #
    def konvolusi2D(self):
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        img1 = self.image
        hasil = konvolusi.konvolusi(img1, kernel)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    #
    # Praktek D2
    #
    def meanFilter(self):
        kernelMean = (1 / 9) * np.array([[1, 1, 1],
                                         [1, 1, 1],
                                         [1, 1, 1]])
        img1 = self.image
        hasil = konvolusi.konvolusi(img1, kernelMean)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    #
    # Praktek D3
    #
    def gaussianFilter(self):
        kernelGaussian = (1.0 / 345) * np.array([
            [1, 5, 7, 5, 1],
            [5, 20, 33, 20, 5],
            [7, 33, 55, 33, 7],
            [5, 20, 33, 20, 5],
            [1, 5, 7, 5, 1]
        ])
        img1 = self.image
        hasil = konvolusi.konvolusi(img1, kernelGaussian)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def imageSharpening(self):
        kernelSharp = (1.0 / 16) * np.array([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ])

        img1 = self.image
        hasil = konvolusi.konvolusi(img1, kernelSharp)

        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def medianFilter(self):
        img1 = cv2.imread('img/1.jpg', cv2.IMREAD_UNCHANGED)
        gray_scale = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        img_out = gray_scale.copy()
        H, W = gray_scale.shape[:2]

        for i in range(3, H - 3):
            for j in range(3, W - 3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = gray_scale[i + k, j + l]
                        neighbors.append(a)
                neighbors.sort()
                median = neighbors[24]
                img_out[i, j] = median

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def maxFilter(self):
        img1 = cv2.imread('img/kidney-diagram.png', cv2.IMREAD_UNCHANGED)
        gray_scale = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        img_out = gray_scale.copy()
        H, W = gray_scale.shape[:2]
        b = 0
        for i in range(3, H - 3):
            for j in range(3, W - 3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = gray_scale[i + k, j + l]
                        neighbors.append(a)
                        if a < b:
                            b = a
                b = max(neighbors)
                img_out.itemset((i, j), b)

        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

        # Pertemuan 5

        #
        # Praktek E1
        #

    def transformasiSmooth(self):
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)

        y += max(y)

        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)
        plt.imshow(img)
        img = cv2.imread('img/1.jpg', 0)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        # ax1
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        # ax2
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        # ax3
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        # ax4
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse fourier')

        plt.show()

        #
        # Praktek E2
        #

    def tranformasiEdge(self):

        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)
        y += max(y)

        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)
        plt.imshow(img)

        img = cv2.imread('img/1.jpg', 0)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.ones((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0

        fshift = dft_shift * mask
        epsilon = 1e-10
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + epsilon)

        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')

        plt.show()

    #
    # Praktek F1
    #
    def deteksiSobel(self):
        img = cv2.imread('img/1.jpg', cv2.IMREAD_GRAYSCALE)

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        sx = konvolusi.konvolusi(img, sobel_x)
        sy = konvolusi.konvolusi(img, sobel_y)

        gradient = np.sqrt((sx * sx) + (sy * sy))

        gradient_normal = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)

        plt.imshow(gradient_normal, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        print(gradient_normal)
        plt.show()

    def deteksiPrewit(self):
        img = cv2.imread('img/1.jpg', cv2.IMREAD_GRAYSCALE)

        sobel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        sobel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        sx = konvolusi.konvolusi(img, sobel_x)
        sy = konvolusi.konvolusi(img, sobel_y)

        gradient = np.sqrt((sx * sx) + (sy * sy))

        gradient_normal = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)

        plt.imshow(gradient_normal, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        print(gradient_normal)
        plt.show()

    # def deteksiRobets(self):
    #     img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    #
    #     sobel_x = np.array([[1, 0], [0 , -1]])
    #     sobel_y = np.array([[0, 1], [-1, 0]])
    #
    #     sx = konvolusi.konvolusi(img, sobel_x)
    #     sy = konvolusi.konvolusi(img, sobel_y)
    #
    #     gradient = np.sqrt((sx * sx) + (sy * sy))
    #
    #     gradient_normal = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
    #
    #     plt.imshow(gradient_normal, cmap='gray', interpolation='bicubic')
    #     plt.xticks([]), plt.yticks([])
    #     plt.show()

    #
    # Praktek F2
    #
    def cannyEdge(self):
        kernelGaussian = (1.0 / 57) * np.array([
            [0, 1, 2, 1, 0],
            [1, 3, 5, 3, 1],
            [2, 5, 9, 5, 2],
            [1, 3, 5, 3, 1],
            [0, 1, 2, 1, 0]
        ])
        img = cv2.imread('img/kidney-diagram.png', cv2.IMREAD_GRAYSCALE)

        hasil = konvolusi.konvolusi(img, kernelGaussian)
        hasil_norm = cv2.normalize(hasil, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("Noise Reduction", hasil_norm)

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        sx = konvolusi.konvolusi(img, sobel_x)
        sy = konvolusi.konvolusi(img, sobel_y)

        theta = np.arctan2(sy, sx)
        cv2.imshow("Finding gradient", theta)

        H, W = hasil.shape
        imgN = np.zeros((H, W), dtype=np.int32)
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = hasil[i, j + 1]
                        r = hasil[i, j - 1]
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = hasil[i + 1, j - 1]
                        r = hasil[i - 1, j + 1]
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = hasil[i + 1, j]
                        r = hasil[i - 1, j]
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = hasil[i - 1, j - 1]
                        r = hasil[i + 1, j + 1]

                    if (hasil[i, j] >= q) and (hasil[i, j] >= r):
                        imgN[i, j] = hasil[i, j]
                    else:
                        imgN[i, j] = 0
                except IndexError as e:
                    pass

        img_N = imgN.astype("uint8")

        cv2.imshow("Non-Maximum Suppresion", img_N)

        weak = 100
        strong = 150
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak):
                    b = weak
                    if (a > strong):
                        b = 255
                else:
                    b = 0
                img_N.itemset((i, j), b)

        img_H1 = img_N.astype("uint8")

        cv2.imshow("hysteresis part 1", img_H1)

        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or (img_H1[i + 1, j] == strong) or (
                                img_H1[i, j - 1] == strong)
                                or (img_H1[i, j + 1] == strong) or (img_H1[i - 1, j] == strong) or (
                                        img_H1[i - 1, j - 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass

        img_H2 = img_H1.astype("uint8")
        cv2.imshow("hysteresis part 2", img_H2)

        # Pertemuan 6

        #
        # Praktek G1
        #

    def morfologi(self):
        img = cv2.imread('img/kidney-diagram.png', cv2.IMREAD_GRAYSCALE)
        ret, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        img_eroded = cv2.erode(img_bin, strel)
        img_dilated = cv2.dilate(img_bin, strel)
        img_opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, strel)
        img_closing = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, strel)

        cv2.imshow("Original Image", img)
        cv2.imshow("Binary Image", img_bin)
        cv2.imshow("Eroded Image", img_eroded)
        cv2.imshow("Dilated Image", img_dilated)
        cv2.imshow("Opening Image", img_opening)
        cv2.imshow("Closing Image", img_closing)

        #
        # Praktek H1
        #

    def thresholdBinary(self):
        img = cv2.imread('img/kidney-diagram.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = 127
        maxval = 255

        ret, thresh1 = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
        cv2.imshow('Binary Thresholding', thresh1)

    def thresholdBinaryInvers(self):
        img = cv2.imread('img/kidney-diagram.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = 127
        maxval = 255

        ret, thresh2 = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY_INV)
        cv2.imshow('Inverse Binary Thresholding', thresh2)

    def thresholdTrunc(self):
        img = cv2.imread('img/kidney-diagram.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = 127
        maxval = 255

        ret, thresh3 = cv2.threshold(gray, thresh, maxval, cv2.THRESH_TRUNC)
        cv2.imshow('Truncated Thresholding', thresh3)

    def thresholdToZero(self):
        img = cv2.imread('img/kidney-diagram.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = 127
        maxval = 255

        ret, thresh4 = cv2.threshold(gray, thresh, maxval, cv2.THRESH_TOZERO)
        cv2.imshow('To Zero Thresholding', thresh4)

    def thresholdToZeroInvers(self):
        img = cv2.imread('img/kidney-diagram.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = 127
        maxval = 255

        ret, thresh5 = cv2.threshold(gray, thresh, maxval, cv2.THRESH_TOZERO_INV)
        cv2.imshow('Inverse To Zero Thresholding', thresh5)

    def thresholdMean(self):
        img = cv2.imread('img/q.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow('Mean Thresholding', imgh)

    def thresholdGaussian(self):
        img = cv2.imread('img/q.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow('Gaussian Thresholding', imgh)

    def thresholdOtsu(self):
        img = cv2.imread('img/q.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = 130
        ret, imgh = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('Otsu Thresholding', imgh)

    def contour(self):
        img = cv2.imread('img/contour.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

            M = cv2.moments(cnt)

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                if abs(w - h) < 10:
                    shape = "Persegi"
                else:
                    shape = "Persegi Panjang"
            elif len(approx) > 4:
                shape = "Lingkaran"
            elif len(approx) == 3:
                shape = "Segitiga"
            elif len(approx) == 10:
                shape = "Bintang"

            cv2.putText(img, shape, (cx - 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(img, (cx, cy), 2, (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 2, (0, 255, 0), 2)

        cv2.imshow("Contour", img)

    #
    # Praktek I1
    #

    def colorTracking(self):
        cam = cv2.VideoCapture(0)
        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_color = np.array([0, 100, 100])
            upper_color = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

    #
    # Praktek I2
    #

    def colorPicker(self):
        def nothing(x):
            pass

        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Trackbars")

        cv2.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
        cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
        cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)



        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            l_h = cv2.getTrackbarPos("L-H", "Trackbars")
            l_s = cv2.getTrackbarPos("L-S", "Trackbars")
            l_v = cv2.getTrackbarPos("L-V", "Trackbars")
            u_h = cv2.getTrackbarPos("U-H", "Trackbars")
            u_s = cv2.getTrackbarPos("U-S", "Trackbars")
            u_v = cv2.getTrackbarPos("U-V", "Trackbars")

            lower_color = np.array([l_h, l_s, l_v])
            upper_color = np.array([u_h, u_s, u_v])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

        #
        # Pertemuan 9
        #

        # Modul I3
    def objectDetection(self):
        cam = cv2.VideoCapture('img/cars1.mp4')
        car_cascade  = cv2.CascadeClassifier('Haar-Cascade/haarcascade_car.xml')

        while True:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # deteksi mobil di video
            cars = car_cascade.detectMultiScale(gray,1.1,3)

            for (x,y,w,h) in cars:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

            cv2.imshow('video', frame)
            if cv2.waitKey(10)&0xFF==ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

    def HOG(self):
        image = data.astronaut()

        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1), visualize=True, channel_axis=-1)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input Image')

        # Rescale gambar untuk tampilan yang lebih baik
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    def HOGPedestrian(self):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        img = cv2.imread("img/pedestrian.jpeg")
        img = imutils.resize(img, width=min(800, img.shape[0]))
        (regions, _) = hog.detectMultiScale(img, winStride=(4, 4), padding=(4, 4), scale=1.05)
        for (x, y, w, h) in regions:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("image", img)
        cv2.waitKey()

    def HaarFaceEye(self):
        # membuat classifier muka
        face_classifier = cv2.CascadeClassifier('Haar-Cascade/haarcascade_frontalface_default.xml')
        eye_classifier = cv2.CascadeClassifier('Haar-Cascade/haarcascade_eye.xml')
        image = cv2.imread('img/churchill1.jpeg')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5, minSize=(30,30))

        if faces is ():
            print('Tidak ada muka yang ditemukan')
        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+w, y+h), (127, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=5, minSize=(10,10))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,255,0), 2)
            cv2.imshow('Face Detection', image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def HaarPedestrian(self):
        # membuat classifier badan
        body_classifier = cv2.CascadeClassifier('Haar-Cascade/haarcascade_fullbody.xml')

        cap = cv2.VideoCapture('img/pedestrian.mp4')
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

            for (x,y,w,h) in bodies:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
                cv2.imshow('Pedestrians', frame)

            if cv2.waitKey(1) == 13:
                break
        cap.release()
        cv2.destroyAllWindows()

    def circleHough(self):
        img = cv2.imread('img/opencv.png',0)
        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=5, maxRadius=0)

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow('detected circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #
    # Pertemuan 10
    #
    def facialLandmark(self):

        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        detector = dlib.get_frontal_face_detector()

        class TooManyFaces(Exception):
            pass

        class NoFaces(Exception):
            pass

        def get_landmarks(im):
            rects = detector(im, 1)
            if len(rects) > 1:
                raise TooManyFaces
            if len(rects) == 0:
                raise NoFaces

            return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

        def annotate_landmarks(im, landmarks):
            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(im, str(idx), pos,
                            fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            fontScale=0.4,
                            color=(0,0, 255)
                            )
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        image = cv2.imread("img/churchill1.jpeg")
        landmarks = get_landmarks(image)
        image_with_landmarks = annotate_landmarks(image, landmarks)

        cv2.imshow("Result", image_with_landmarks)
        cv2.imwrite("image_with_landmarks.jpg", image_with_landmarks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def swapFace(self):
        SCALE_FACTOR = 2
        FEATHER_AMOUNT = 11
        FACE_POINTS = list(range(17, 68))
        MOUTH_POINTS = list(range(48, 61))
        RIGHT_BROW_POINTS = list(range(17, 22))
        LEFT_BROW_POINTS = list(range(22, 27))
        RIGHT_EYE_POINTS = list(range(36, 42))
        LEFT_EYE_POINTS = list(range(42, 48))
        NOSE_POINTS = list(range(27, 35))
        JAW_POINTS = list(range(0, 17))

        ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                        RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

        OVERLAY_POINTS = [
            LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
            NOSE_POINTS + MOUTH_POINTS
        ]

        COLOUR_CORRECT_BLUR_FRAC = 0.6

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)

        class TooManyFaces(Exception):
            pass

        class NoFaces(Exception):
            pass

        def get_landmarks(im):
            rects = detector(im, 1)

            if len(rects) > 1:
                raise TooManyFaces
            if len(rects) == 0:
                raise NoFaces

            return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

        def annotate_landmarks(im, landmarks):

            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(im, str(idx), pos,
                            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            fontScale=0.4,
                            color=(0, 0, 255)
                            )
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        def draw_convex_hull(im, points, color):
            points = cv2.convexHull(points)
            cv2.fillConvexPoly(im, points, color=color)

        def get_face_mask(im, landmarks):
            im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

            for group in OVERLAY_POINTS:
                draw_convex_hull(im, landmarks[group], color=1)

            im = numpy.array([im, im, im]).transpose((1, 2, 0))
            im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
            im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

            return im

        def transformation_from_points(points1, points2):
            points1 = points1.astype(numpy.float64)
            points2 = points2.astype(numpy.float64)
            c1 = numpy.mean(points1, axis=0)
            c2 = numpy.mean(points2, axis=0)
            points1 -= c1
            points2 -= c2
            s1 = numpy.std(points1)
            s2 = numpy.std(points2)
            points1 /= s1
            points2 /= s2
            U, S, Vt = numpy.linalg.svd(points1.T * points2)
            R = (U * Vt).T
            return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                               c2.T - (s2 / s1) * R * c1.T)),
                                 numpy.matrix([0., 0., 1.])])

        def read_im_and_landmarks(image):
            im = image
            im = cv2.resize(im, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                                 im.shape[0] * SCALE_FACTOR))
            s = get_landmarks(im)
            return im, s

        def warp_im(im, M, dshape):
            output_im = numpy.zeros(dshape, dtype=im.dtype)
            cv2.warpAffine(im,
                           M[:2],
                           (dshape[1], dshape[0]),
                           dst=output_im,
                           borderMode=cv2.BORDER_TRANSPARENT,
                           flags=cv2.WARP_INVERSE_MAP)
            return output_im

        def correct_colours(im1, im2, landmarks1):
            blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
            blur_amount = int(blur_amount)
            if blur_amount % 2 == 0:
                blur_amount += 1
            im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
            im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
            # Avoid divide-by-zero errors.
            im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
            return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                    im2_blur.astype(numpy.float64))

        def swappy(image1, image2):
            im1, landmarks1 = read_im_and_landmarks(image1)
            im2, landmarks2 = read_im_and_landmarks(image2)
            M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                           landmarks2[ALIGN_POINTS])
            mask = get_face_mask(im2, landmarks2)
            warped_mask = warp_im(mask, M, im1.shape)
            combined_mask = numpy.max([get_face_mask(im1, landmarks1),
                                       warped_mask],
                                      axis=0)
            warped_im2 = warp_im(im2, M, im1.shape)
            warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
            output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
            cv2.imwrite('output.jpg', output_im)
            image = cv2.imread('output.jpg')
            return image




        ## Enter the paths to your input images here
        image1 = cv2.imread('img/haaland.jpeg')
        image2 = cv2.imread('img/churchill1.jpeg')
        swapped = swappy(image1, image2)
        cv2.imshow('Face Swap 1', swapped)
        swapped = swappy(image2, image1)
        cv2.imshow('Face Swap 2', swapped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def swapFaceRealTime(self):
        SCALE_FACTOR = 1
        FEATHER_AMOUNT = 11
        FACE_POINTS = list(range(17, 68))
        MOUTH_POINTS = list(range(48, 61))
        RIGHT_BROW_POINTS = list(range(17, 22))
        LEFT_BROW_POINTS = list(range(22, 27))
        RIGHT_EYE_POINTS = list(range(36, 42))
        LEFT_EYE_POINTS = list(range(42, 48))
        NOSE_POINTS = list(range(27, 35))
        JAW_POINTS = list(range(0, 17))

        ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                        RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

        OVERLAY_POINTS = [
            LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
            NOSE_POINTS + MOUTH_POINTS
        ]

        COLOUR_CORRECT_BLUR_FRAC = 0.6
        cascade_path = "Haar-Cascade/haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)

        def get_landmarks(im, dlibOn):
            rects = detector(im, 1)

            if len(rects) > 1:
                raise TooManyFaces
            if len(rects) == 0:
                raise NoFaces

            return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

        class TooManyFaces(Exception):
            pass

        class NoFaces(Exception):
            pass

        def annotate_landmarks(im, landmarks):

            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(im, str(idx), pos,
                            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            fontScale=0.4,
                            color=(0, 0, 255)
                            )
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        def draw_convex_hull(im, points, color):
            points = cv2.convexHull(points)
            cv2.fillConvexPoly(im, points, color=color)

        def get_face_mask(im, landmarks):
            im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

            for group in OVERLAY_POINTS:
                draw_convex_hull(im, landmarks[group], color=1)

            im = numpy.array([im, im, im]).transpose((1, 2, 0))
            im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
            im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

            return im

        def transformation_from_points(points1, points2):
            points1 = points1.astype(numpy.float64)
            points2 = points2.astype(numpy.float64)
            c1 = numpy.mean(points1, axis=0)
            c2 = numpy.mean(points2, axis=0)
            points1 -= c1
            points2 -= c2
            s1 = numpy.std(points1)
            s2 = numpy.std(points2)
            points1 /= s1
            points2 /= s2
            U, S, Vt = numpy.linalg.svd(points1.T * points2)
            R = (U * Vt).T
            return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])

        def read_im_and_landmarks(fname):
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = cv2.resize(im, None, fx=0.35, fy=0.35,
                            interpolation=cv2.INTER_LINEAR)
            im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                                 im.shape[0] * SCALE_FACTOR))
            s = get_landmarks(im, dlibOn)
            return im, s

        def warp_im(im, M, dshape):
            output_im = numpy.zeros(dshape, dtype=im.dtype)
            cv2.warpAffine(im,
                           M[:2],
                           (dshape[1], dshape[0]),
                           dst=output_im,
                           borderMode=cv2.BORDER_TRANSPARENT,
                           flags=cv2.WARP_INVERSE_MAP)
            return output_im

        def correct_colours(im1, im2, landmarks1):
            blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
            blur_amount = int(blur_amount)
            if blur_amount % 2 == 0:
                blur_amount += 1
            im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
            im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
            # Avoid divide-by-zero errors.
            im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
            return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                    im2_blur.astype(numpy.float64))

        def face_swap(img, name):
            s = get_landmarks(img, True)
            if (s == "error"):
                print("No or too many faces")
            return img
            im1, landmarks1 = img, s
            im2, landmarks2 = read_im_and_landmarks(name)
            M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                           landmarks2[ALIGN_POINTS])
            mask = get_face_mask(im2, landmarks2)
            warped_mask = warp_im(mask, M, im1.shape)
            combined_mask = numpy.max([get_face_mask(im1, landmarks1),
                                       warped_mask],
                                      axis=0)
            warped_im2 = warp_im(im2, M, im1.shape)
            warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)


            output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
            # output_im is no longer in the expected OpenCV format so we use openCV
            # to write the image to diks and then reload it
            cv2.imwrite('output.jpg', output_im)
            image = cv2.imread('output.jpg')
            frame = cv2.resize(image, None, fx=1.5, fy=1.5,
                               interpolation=cv2.INTER_LINEAR)
            return image


        cap = cv2.VideoCapture(0)
        # Name is the image we want to swap onto ours
        # dlibOn controls if use dlib's facial landmark detector (better)
        # or use HAAR Cascade Classifiers (faster)
        filter_image = "img/xi.jpeg"  ### Put your image here!
        dlibOn = True
        while True:
            ret, frame = cap.read()
            # Reduce image size by 75% to reduce processing time and improve

            frame = cv2.resize(frame, None, fx=0.75, fy=0.75,
                           interpolation=cv2.INTER_LINEAR)
        # flip image so that it's more mirror like
            frame = cv2.flip(frame, 1)
            cv2.imshow('Our Amazing Face Swapper', face_swap(frame, filter_image))
            if cv2.waitKey(1) == 13:
                break
        cap.release()
        cv2.destroyAllWindows()

    def yawnDetection(self):
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        detector = dlib.get_frontal_face_detector()

        def get_landmarks(im):
            rects = detector(im, 1)
            if len(rects) > 1:
                return "error"
            if len(rects) == 0:
                return "error"
            return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

        def annotate_landmarks(im, landmarks):
            im = im.copy()
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4,
                            color=(0, 0, 255))
                cv2.circle(im, pos, 3, color=(0, 255, 255))
            return im

        def top_lip(landmarks):
            top_lip_pts = []
            for i in range(50, 53):
                top_lip_pts.append(landmarks[i])
            for i in range(61, 64):
                top_lip_pts.append(landmarks[i])
            top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
            top_lip_mean = np.mean(top_lip_pts, axis=0)
            return int(top_lip_mean[:, 1])

        def bottom_lip(landmarks):
            bottom_lip_pts = []
            for i in range(65, 68):
                bottom_lip_pts.append(landmarks[i])
            for i in range(56, 59):
                bottom_lip_pts.append(landmarks[i])
            bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
            bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
            return int(bottom_lip_mean[:, 1])

        def mouth_open(image):
            landmarks = get_landmarks(image)
            if landmarks == "error":
                return image, 0
            image_with_landmarks = annotate_landmarks(image, landmarks)
            top_lip_center = top_lip(landmarks)
            bottom_lip_center = bottom_lip(landmarks)
            lip_distance = abs(top_lip_center - bottom_lip_center)
            return image_with_landmarks, lip_distance

        cap = cv2.VideoCapture(0)
        yawns = 0
        yawn_status = False

        while True:
            ret, frame = cap.read()
            image_landmarks, lip_distance = mouth_open(frame)
            prev_yawn_status = yawn_status

            if lip_distance > 25:
                yawn_status = True
                cv2.putText(frame, "Subject is Yawning", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                output_text = " Yawn Count: " + str(yawns + 1)
                cv2.putText(frame, output_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

            else:
                yawn_status = False

            if prev_yawn_status == True and yawn_status == False:
                yawns += 1

            cv2.imshow('Live Landmarks', image_landmarks)
            cv2.imshow('Yawn Detection', frame)

            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break

        cap.release()
        cv2.destroyAllWindows()

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:  # row[0],col[1],channel[2]
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0],
                     self.image.strides[0], qformat)

        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))

            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)
        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 3')
window.show()
sys.exit(app.exec_())
