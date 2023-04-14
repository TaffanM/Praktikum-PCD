import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
import math
from matplotlib import pyplot as plt
import konvolusi

import test


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

    # Pertemuan 2

    #
    # Praktek A2
    #
    def loadClicked(self):
        self.image = cv2.imread('kidney-diagram.png')
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
        image1 = cv2.imread('01.png', 1)
        image2 = cv2.imread('1.jpg', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        and_op = cv2.bitwise_and(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi AND', and_op)
        cv2.waitKey(0)

    def operasiOR(self):
        image1 = cv2.imread('01.png', 1)
        image2 = cv2.imread('1.jpg', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        or_op = cv2.bitwise_or(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi OR', or_op)
        cv2.waitKey(0)

    def operasiXOR(self):
        image1 = cv2.imread('01.png', 1)
        image2 = cv2.imread('1.jpg', 1)
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
        image1 = cv2.imread('01.png', 0)
        image2 = cv2.imread('1.jpg', 0)
        image_plus = image1 + image2
        image_minus = image1 - image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Tambah', image_plus)
        cv2.imshow('Image Kurang', image_minus)
        cv2.waitKey(0)

    def aritmatikaKali(self):
        image1 = cv2.imread('01.png', 0)
        image2 = cv2.imread('1.jpg', 0)
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
        img1 = cv2.imread('1.jpg', cv2.IMREAD_UNCHANGED)
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
        img1 = cv2.imread('kidney-diagram.png', cv2.IMREAD_UNCHANGED)
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
        img = cv2.imread('1.jpg', 0)
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

        img = cv2.imread('1.jpg', 0)
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
        img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

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
        img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

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
        img = cv2.imread('kidney-diagram.png', cv2.IMREAD_GRAYSCALE)

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
        img = cv2.imread('kidney-diagram.png', cv2.IMREAD_GRAYSCALE)
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
        img = cv2.imread('kidney-diagram.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = 127
        maxval = 255

        ret, thresh1 = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
        cv2.imshow('Binary Thresholding', thresh1)

    def thresholdBinaryInvers(self):
        img = cv2.imread('kidney-diagram.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = 127
        maxval = 255

        ret, thresh2 = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY_INV)
        cv2.imshow('Inverse Binary Thresholding', thresh2)

    def thresholdTrunc(self):
        img = cv2.imread('kidney-diagram.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = 127
        maxval = 255

        ret, thresh3 = cv2.threshold(gray, thresh, maxval, cv2.THRESH_TRUNC)
        cv2.imshow('Truncated Thresholding', thresh3)

    def thresholdToZero(self):
        img = cv2.imread('kidney-diagram.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = 127
        maxval = 255

        ret, thresh4 = cv2.threshold(gray, thresh, maxval, cv2.THRESH_TOZERO)
        cv2.imshow('To Zero Thresholding', thresh4)

    def thresholdToZeroInvers(self):
        img = cv2.imread('kidney-diagram.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = 127
        maxval = 255

        ret, thresh5 = cv2.threshold(gray, thresh, maxval, cv2.THRESH_TOZERO_INV)
        cv2.imshow('Inverse To Zero Thresholding', thresh5)

    def thresholdMean(self):
        img = cv2.imread('q.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow('Mean Thresholding', imgh)

    def thresholdGaussian(self):
        img = cv2.imread('q.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow('Gaussian Thresholding', imgh)

    def thresholdOtsu(self):
        img = cv2.imread('q.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = 130
        ret, imgh = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('Otsu Thresholding', imgh)

    def contour(self):
        img = cv2.imread('contour.jpg')
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
