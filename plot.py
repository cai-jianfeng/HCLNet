"""
created on:2023/3/5 10:52
@author:caijianfeng
"""

import xlrd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

colors = []


class plot_mode:
    def __init__(self, mode='ground'):
        """
        :param mode: ground: ground truth;
                     predict: predict.
        """
        self.mode = mode
    
    def plot_labels(self, label_path=None, color_path=None, label_pic_name=None):
        if color_path:
            color_sheets = xlrd.open_workbook(color_path)
            color_sheet = color_sheets.sheet_by_index(0)
            rows = color_sheet.nrows
            # cols = color_sheet.ncols
            for i in range(rows):
                color = [color_sheet.cell_value(i, 0), color_sheet.cell_value(i, 1), color_sheet.cell_value(i, 2)]
                colors.append(color)
            print('color mat load succeed!')
        else:
            print('color_path is necessary!')
            return
        if label_path and label_pic_name:
            labels_sheets = xlrd.open_workbook(label_path)
            labels_sheet = labels_sheets.sheet_by_index(0)
            rows = labels_sheet.nrows
            cols = labels_sheet.ncols
            print('label load succeed!')
            R = np.zeros((rows, cols), dtype='uint8')
            G = np.zeros((rows, cols), dtype='uint8')
            B = np.zeros((rows, cols), dtype='uint8')
            rs = tqdm(range(rows))
            for i in rs:
                for j in range(cols):
                    label = int(labels_sheet.cell_value(i, j))
                    R[i][j] = colors[label][0]
                    G[i][j] = colors[label][1]
                    B[i][j] = colors[label][2]
                rs.desc = 'plot -> row: {}/{}'.format(i, rows)
            
            label_map = cv2.merge([B, G, R])
            # cv2.imshow('label', label_map)
            # cv2.waitKey(0)
            save_path = label_pic_name
            cv2.imwrite(save_path, label_map)
        else:
            print('label_path/label_pic_name is necessary!')
            return
    
    def plt_image(self, title, x_data, y_data, label, xlabel, ylabel, save_path, color='r'):
        """
        Plot line charts for loss, accuracy, and so on
        :param xlabel: horizontal axis --> str
        :param ylabel: vertical coordinates --> str
        :param label: plot label --> str
        :param color: Color of polyline --> str
        :param title: Image caption --> str
        :param x_data: X-axis data --> list
        :param y_data: Y-axis data --> list
        :param save_path: save path --> str
        :return:
        """
        plt.clf()  # Empty the previous image
        plt.title(title)
        plt.grid(linestyle=":")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x_data, y_data, color, label=label)
        plt.tight_layout()  # Subplot parameters are automatically adjusted to fill the entire image area
        plt.savefig(save_path)
        plt.show()

    def plot_mask_labels(self, patch_size=None, label_path=None, predict_path=None, color_path=None, label_pic_name=None):
        if patch_size is None:
            patch_size = [15, 15]
        if not label_path or not predict_path or not color_path or not label_pic_name:
            print('Necessary input parameters are missing!')
            return

        color_sheets = xlrd.open_workbook(color_path)
        color_sheet = color_sheets.sheet_by_index(0)
        rows = color_sheet.nrows
        # cols = color_sheet.ncols
        for i in range(rows):
            color = [color_sheet.cell_value(i, 0), color_sheet.cell_value(i, 1), color_sheet.cell_value(i, 2)]
            colors.append(color)
        print('color mat load succeed!')
        labels_sheets = xlrd.open_workbook(label_path)
        labels_sheet = labels_sheets.sheet_by_index(0)
        predict_sheets = xlrd.open_workbook(predict_path)
        predict_sheet = predict_sheets.sheet_by_index(0)
        rows = predict_sheet.nrows
        cols = predict_sheet.ncols
        print('rows:{};cols:{}'.format(rows, cols))
        print('label/predict load succeed!')
        R = np.zeros((rows, cols), dtype='uint8')
        G = np.zeros((rows, cols), dtype='uint8')
        B = np.zeros((rows, cols), dtype='uint8')
        rs = tqdm(range(rows))
        for i in rs:
            for j in range(cols):
                label = int(labels_sheet.cell_value(i + patch_size[0] // 2, j + patch_size[1] // 2))
                predict = int(predict_sheet.cell_value(i, j))
                if label != 0:
                    R[i][j] = colors[predict][0]
                    G[i][j] = colors[predict][1]
                    B[i][j] = colors[predict][2]
            rs.desc = 'plot -> row: {}/{}'.format(i, rows)

        label_map = cv2.merge([B, G, R])
        save_path = label_pic_name
        cv2.imwrite(save_path, label_map)
