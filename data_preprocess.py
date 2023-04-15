"""
created on:2022/11/13 14:45
@author:caijianfeng
"""
import xlrd
import numpy as np
from tqdm import tqdm
import json
import os
import xlsxwriter
import random
from feature_extract import feature


class Data:
    def __init__(self):
        self.data_sets = None
        self.label_set = None
        self.feature = feature()

    def get_T_data(self, data_path, patch_size):
        """
        返回每个数据点的T矩阵
        :param patch_size: 对比网络中的另一个网络的二维数据的大小(rows, cols)
        :param data_path: T矩阵的实部数据位置和虚部数据位置(list)
        :return: 所有数据点的T矩阵(data_num, 3, 3) -> numpy
        """
        data_path_R, data_path_I = data_path  # T矩阵的实部和虚部的路径
        sheet_book_R = xlrd.open_workbook(filename=data_path_R)  # 打开T矩阵实部excel文件
        sheet_book_I = xlrd.open_workbook(filename=data_path_I)
        sheet_nums = sheet_book_R.nsheets  # 获取excel文件的sheet数目
        rows = sheet_book_R.sheet_by_index(0).nrows  # 获取数据行数
        cols = sheet_book_R.sheet_by_index(0).ncols  # 获取数据列数
        print(rows, ';', cols)
        data_num = (rows - patch_size[0]) * (cols - patch_size[1])  # 数据总数=(行数 - 块行) x (列数 - 块列)
        data_set = []  # 总数据(data_nums, rows(3), cols(3))
        data_nums = tqdm(range(data_num))
        for i in data_nums:
            data = np.zeros((3, 3), dtype='complex')  # 每个数据点的T矩阵
            for sheet_num in range(sheet_nums):
                sheet_R = sheet_book_R.sheet_by_index(sheet_num)
                sheet_I = sheet_book_I.sheet_by_index(sheet_num)
                data[sheet_num // 3, sheet_num % 3] = sheet_R.cell_value(
                    (i // (rows - patch_size[0])) + (patch_size[0] // 2),
                    (i % (cols - patch_size[1])) + (patch_size[1] // 2)) \
                                                      + sheet_I.cell_value(
                    (i // (rows - patch_size[0])) + (patch_size[0] // 2),
                    (i % (cols - patch_size[1])) + (patch_size[1] // 2)) * 1j  # 将实部与虚部结合为复数
            data_set.append(data)

        return np.array(data_set)

    def get_feature_data_by_T(self, data_path, patch_size):
        """
        获取每个数据点的特征分解值
        :param patch_size: 对比网络中的另一个网络的二维数据的大小(rows, cols)
        :param data_path: T矩阵的实部数据位置和虚部数据位置(list)
        :return: 所有数据点的特征值集合(data_nums, feature_nums) -> numpy
        """
        feature_util = self.feature  # 特征提取函数
        # feature_util = feature()
        features = []  # 总数据的特征集合(data_nums, feature_nums)
        data_set = self.get_T_data(data_path=data_path, patch_size=patch_size)  # 获取总数据的T矩阵(data_nums, 3, 3)
        for data in data_set:  # 对于每个数据点, 提取特征集合
            eig1, eig2, eig3 = feature_util.eigen_decomposition(T=data)
            H, aerfa, Ani = feature_util.H_aerfa_Ani_decomposition(T=data)
            RF1, RF2, RF3, RF4, RF5, RF6 = feature_util.RF_decomposition(T=data)
            Pv_F, Pd_F, Ps_F = feature_util.Freeman_decomposition(T=data)
            Ps_H, Pd_H, Pr_H = feature_util.Holm_decomposition(T=data)
            Srl, Srr, Sll = feature_util.krogager_decomposition(T=data)
            Ph_k, Pmd_k, Pcd_k, Pod_k, Ps_k, Pd_k, Pv_k = feature_util.seven_component_scattering_power_decomposition(
                T=data)
            data_feature = np.array([eig1, eig2, eig3, H, aerfa, Ani, RF1, RF2, RF3, RF4, RF5, RF6, Pv_F, Pd_F, Ps_F,
                                     Ps_H, Pd_H, Pr_H, Srl, Srr, Sll, Ph_k, Pmd_k, Pcd_k, Pod_k, Ps_k, Pd_k, Pv_k])
            features.append(data_feature)
        features = np.array(features)
        return features

    def get_feature_data_by_T_21(self, data_path, patch_size):
        """
        获取每个数据点的特征分解值(21个)
        :param patch_size: 对比网络中的另一个网络的二维数据的大小(rows, cols)
        :param data_path: T矩阵的实部数据位置和虚部数据位置(list)
        :return: 所有数据点的特征值集合(data_nums, feature_nums) -> numpy
        """
        feature_util = self.feature  # 特征提取函数
        # feature_util = feature()
        features = []  # 总数据的特征集合(data_nums, feature_nums)
        data_set = self.get_T_data(data_path=data_path, patch_size=patch_size)  # 获取总数据的T矩阵(data_nums, 3, 3)
        for data in data_set:  # 对于每个数据点, 提取特征集合
            eig1, eig2, eig3 = feature_util.eigen_decomposition(T=data)
            H, aerfa, Ani = feature_util.H_aerfa_Ani_decomposition(T=data)
            RF1, RF2, RF3, RF4, RF5, RF6 = feature_util.RF_decomposition(T=data)
            Pv_F, Pd_F, Ps_F = feature_util.Freeman_decomposition(T=data)
            Ps_H, Pd_H, Pr_H = feature_util.Holm_decomposition(T=data)
            Srl, Srr, Sll = feature_util.krogager_decomposition(T=data)
            data_feature = np.array([eig1, eig2, eig3, H, aerfa, Ani, RF1, RF2, RF3, RF4, RF5, RF6,
                                     Pv_F, Pd_F, Ps_F, Ps_H, Pd_H, Pr_H, Srl, Srr, Sll])
            features.append(data_feature)
        features = np.array(features)
        return features

    def get_data_list(self, data_path):
        """
        Retrieve the data for the specified path
        :param data_path: data path --> str
        :return: data(three dimensions:[channel, row, column]) --> numpy array
        """
        sheets = xlrd.open_workbook(data_path)  # Open the Excel file for data_path
        sheets_num = sheets.nsheets  # Gets the number of sheets in an Excel file
        # sheets_names = sheets.sheet_names() # Get the names of all sheets
        data_sets = []  # three-dimensional data(channel, rows, cols)
        for num in tqdm(range(sheets_num)):  # For each sheet
            sheet = sheets.sheet_by_index(num)  # Get the sheet
            rows = sheet.nrows  # Gets the number of rows in the sheet
            columns = sheet.ncols  # Gets the number of columns in the sheet
            data_set = [[0 for _ in range(columns)] for _ in range(rows)]  # two-dimensional data(rows, cols)
            for row in range(rows):
                for column in range(columns):
                    # cell = sheet.cell(row, column)  # Getting cells
                    # data_flevoland4 = cell.value  # Get the cell data
                    data = sheet.cell_value(row, column)  # Get the cell data
                    data_set[row][column] = data
            data_sets.append(data_set)
        data_sets = np.array(data_sets)
        return data_sets

    def get_label_list(self, label_path):
        """
        获取指定路径的标签集
        :param label_path: 标签的path --> str
        :return: 标签集(二维数据:[row, column]) --> numpy array
        """
        sheets = xlrd.open_workbook(label_path)
        sheet = sheets.sheet_by_index(0)
        rows = sheet.nrows  # 获取sheet页的行数
        columns = sheet.ncols  # 获取sheet页的列数
        label_set = [[0 for _ in range(columns)] for _ in range(rows)]
        for row in tqdm(range(rows)):
            for column in range(columns):
                label = sheet.cell_value(row, column)  # 获取单元格数据
                label_set[row][column] = label

        label_set = np.array(label_set)
        # print(label_set.shape)  # 读取Flevoland4数据为:1400 * 1200
        # self.label_set = label_set
        return label_set

    def get_data(self, data_paths, dim):
        """
        获取预处理好的数据集(dim[0], dim[1], channels, rows, cols)
        每个数据点是以自身为中心的长宽为(rows, cols)的patch
        :param dim: 图像的长宽 --> tuple (row, column)
        :param data_paths:预处理好的数据集路径 --> list
        :return: list (n维数据)
        """
        datas = []  # n维数据: (dim[0], dim[1], channels, rows, cols)
        num = 0
        rs = tqdm(range(dim[0]))
        for row in rs:
            data_sets = []
            for column in range(dim[1]):
                data_path = data_paths[num]
                data_set = self.get_data_list(data_path=data_path)  # (channels, rows, cols)
                data_sets.append(data_set)
                num += 1
                rs.desc = 'row: {}/{}; file:'.format(row, dim[0]) + data_path
            datas.append(data_sets)
        datas = np.array(datas).astype('float32')
        return datas

    def save_feature_data(self, data_path, patch_size, feature_data_save_path, contrastive_list_path):
        """
        保存数据集的特征集合(所有数据点一个excel文件, 每个数据点一个sheet)
        :param patch_size: 对比网络中的另一个网络的二维数据的大小(rows, cols)
        :param data_path: T矩阵的实部数据位置和虚部数据位置(list)
        :param feature_data_save_path: 所有数据特征集合存放路径的父目录
        :param contrastive_list_path: 特征集合说明文件
        :return: None
        """
        datas_features = self.get_feature_data_by_T(data_path=data_path,
                                                    patch_size=patch_size)  # (data_nums, feature_nums)
        contrastive_list = []
        feature_save_path = os.path.join(feature_data_save_path, 'features.xlsx')  # 所有数据的特征集合存放在一个excel文件中
        contrastive_list.append(feature_save_path)
        book = xlsxwriter.Workbook(filename=feature_save_path)
        for i, data_features in enumerate(datas_features):
            # 每个数据点的sheet命名为:sheet_i
            sheet = book.add_worksheet('sheet_{}'.format(i))
            # sheet.write_row(data_features)
            for j, single_feature in enumerate(data_features):
                sheet.write(0, j, single_feature)
        book.close()
        with open(contrastive_list_path, 'a') as f:  # 写入特征集合路径
            for path in contrastive_list:
                f.write(path)
        print('数据点特征集合生成完毕！')

    def get_feature_data_by_file(self, feature_data_path, sheet_num):
        """
        根据索引与特征集合存放路径获取数据点特征集合
        :param feature_data_path: 特征集合存放路径 -> str
        :param sheet_num: 所需数据的位置(第几个数据)
        :return: 指定数据点的特征集合
        """
        book = xlrd.open_workbook(filename=feature_data_path)
        sheet = book.sheet_by_index(sheet_num)
        data_feature = []
        ncol = sheet.ncols
        for col in range(ncol):
            data_feature.append(sheet.cell_value(0, col))
        data_feature = np.array(data_feature)
        return data_feature

    def get_feature_data_by_file_28(self, feature_data_path, sheet_num):
        """
        根据索引与特征集合存放路径获取数据点特征集合(21个)
        :param feature_data_path: 特征集合存放路径 -> str
        :param sheet_num: 所需数据的位置(第几个数据)
        :return: 指定数据点的特征集合
        """
        book = xlrd.open_workbook(filename=feature_data_path)
        sheet = book.sheet_by_index(sheet_num)
        data_feature = []
        ncol = 21
        for col in range(ncol):
            data_feature.append(sheet.cell_value(0, col))
        data_feature = np.array(data_feature)
        return data_feature

    def save_data_label_segmentation_TRI(self, data_path, label_path, target_path, contrastive_path, train_list_path,
                                         eval_list_path, patch_size, predict_list_path):
        """
        Dataset segmentation (the data of a whole image is cut into patch size and saved), and the segmented data set is the real part + imaginary part of T matrix
        :param contrastive_path: CL set specification file
        :param predict_list_path: Prediction set specification file --> str
        :param data_path: Path of original data (whole image) (real part, imaginary part) --> tuple
        :param label_path: Path to the original label (whole image) --> str
        :param target_path: Where to save the split dataset (folder) --> str
        :param train_list_path: train set specification file --> str
        :param eval_list_path: test set specification file --> str
        :param patch_size: The size of the chunk to split([row, column]) --> tuple
        :return: None
        """
        data_R_path = data_path[0]  # Real part of the dataset path
        data_I_path = data_path[1]  # Imaginary part of the dataset path
        data_R_sets = self.get_data_list(data_path=data_R_path)  # (channels, rows, cols)
        data_I_sets = self.get_data_list(data_path=data_I_path)  # (channels, rows, cols)
        label_set = self.get_label_list(label_path=label_path)  # (rows, cols)
        dim = data_R_sets.shape  # data dimension:(channels, rows, columns)
        print('dim:', dim)
        sum_contrastive_list, sum_train_list, sum_eval_list, sum_predict_list = 0, 0, 0, 0  # Amount of data in contrastive, train, test, and prediction sets
        contrastive_list = []  # CL data
        train_list = []  # train data
        eval_list = []  # test data
        predict_list = []  # predict data
        data_detail_list = {}  # Data information description
        for i in tqdm(range(0, dim[1] - patch_size[0])):
            for j in range(0, dim[2] - patch_size[1]):
                # Store partition T matrix data set path, path name :TRIX.xlsx -> X represents the XTH data block (top left to bottom right)
                # The structure is:
                #   sheet_RX:Represents the real part of the x-th position of the T matrix (top left to bottom right)
                #   sheet_IX:Represents the imaginary part of the x-th position of the T matrix
                target_paths = os.path.join(target_path, 'TRI' + str(sum_predict_list) + '.xlsx')
                if not os.path.exists(
                        target_paths):  # If the datapath already exists, the data has been generated and is skipped
                    book = xlsxwriter.Workbook(
                        filename=target_paths)  # Generates an Excel file with the specified filename
                    for channel in range(dim[0]):  # For each channel (number of real parts (9))
                        sheet = book.add_worksheet(
                            'sheet_R' + str(channel))  # Adds a sheet with the specified sheet name
                        # Writes the real part of the data for the specified block size
                        for row in range(patch_size[0]):
                            for column in range(patch_size[1]):
                                sheet.write(row, column, data_R_sets[channel][i + row][j + column])
                    for channel in range(dim[0]):  # For each channel (number of imaginary parts (9))
                        sheet = book.add_worksheet('sheet_I' + str(channel))
                        # Writes the imaginary part of the data for the specified block size
                        for row in range(patch_size[0]):
                            for column in range(patch_size[1]):
                                sheet.write(row, column, data_I_sets[channel][i + row][j + column])
                    book.close()
                # Convert the position of the corresponding label
                label = label_set[i + patch_size[0] // 2][j + patch_size[1] // 2]
                # print(num, ':', label)
                if random.random() < 0.1 and label != 0:  # The CL data set is taken to be 10% of the total data
                    contrastive_list.append(target_paths + '\t%d' % label + '\n')
                    sum_contrastive_list += 1
                if random.random() < 0.01 and label != 0:  # The training data set is taken to be 1% of the total data
                    train_list.append(target_paths + '\t%d' % label + '\n')
                    sum_train_list += 1
                if random.random() > 0.9 and label != 0:  # The test data set is taken to be 10% of the total data
                    eval_list.append(target_paths + '\t%d' % label + '\n')
                    sum_eval_list += 1
                predict_list.append(
                    target_paths + '\t%d' % label + '\n')  # The whole data is used as the prediction set
                sum_predict_list += 1

        random.shuffle(contrastive_list)
        with open(contrastive_path, 'a') as f:  # Write CL set information
            for contrastive_data in contrastive_list:
                f.write(contrastive_data)
        random.shuffle(eval_list)  # 打乱测试集
        with open(eval_list_path, 'a') as f:  # Write test set information
            for eval_data in eval_list:
                f.write(eval_data)
        random.shuffle(train_list)  # 打乱训练集
        with open(train_list_path, 'a') as f:  # Write train set information
            for train_data in train_list:
                f.write(train_data)
        with open(predict_list_path, 'a') as f:  # Write predict set information
            for predict_data in predict_list:
                f.write(predict_data)
        data_detail_list['data_list_path'] = target_path  # Partitioned parent directory of data
        data_detail_list['dim'] = [dim[1] - patch_size[0], dim[2] - patch_size[1]]  # Total dataset size(rows, cols)
        data_detail_list['contrastive_num'] = sum_contrastive_list  # Amount of CL data
        data_detail_list['train_num'] = sum_train_list  # Amount of training data
        data_detail_list['eval_num'] = sum_eval_list  # Amount of test data
        data_detail_list['predict_num'] = sum_predict_list  # 预测数据量
        jsons = json.dumps(data_detail_list, sort_keys=True, indent=4, separators=(',', ':'))  # Convert the dictionary to json
        with open(os.path.join(target_path, 'readme.json'), 'w') as f:  # Writing data information
            f.write(jsons)

    def save_data_label_segmentation_TRI_superpixel(self, superpixel_json_path, label_path, train_list_path,
                                                    eval_list_path):
        pass

    def get_feature_by_T_metric(self, data):
        eig1, eig2, eig3 = self.feature.eigen_decomposition(T=data)
        H, aerfa, Ani = self.feature.H_aerfa_Ani_decomposition(T=data)
        RF1, RF2, RF3, RF4, RF5, RF6 = self.feature.RF_decomposition(T=data)
        Pv_F, Pd_F, Ps_F = self.feature.Freeman_decomposition(T=data)
        Ps_H, Pd_H, Pr_H = self.feature.Holm_decomposition(T=data)
        Srl, Srr, Sll = self.feature.krogager_decomposition(T=data)
        Ph_k, Pmd_k, Pcd_k, Pod_k, Ps_k, Pd_k, Pv_k = self.feature.seven_component_scattering_power_decomposition(
            T=data)
        data_feature = np.array([eig1, eig2, eig3, H, aerfa, Ani, RF1, RF2, RF3, RF4, RF5, RF6, Pv_F, Pd_F, Ps_F,
                                 Ps_H, Pd_H, Pr_H, Srl, Srr, Sll, Ph_k, Pmd_k, Pcd_k, Pod_k, Ps_k, Pd_k, Pv_k])
        return data_feature

    def save_contrastive_data(self, data_path, target_path, contrastive_list_path, patch_size):
        """
        保存对比学习训练数据
        readme.json文件保存在二维数据块位置的父目录的父目录中
        :param data_path: T矩阵的实部数据位置和虚部数据位置(list)
        :param target_path: 分割后的数据保存位置(list), 包括二维数据块位置与一维特征集位置
        :param contrastive_list_path: 对比训练集说明文件 --> str
        :param patch_size: 切分的数据块的大小([row, column]) --> tuple
        :return: None
        """
        datas_features = self.get_feature_data_by_T(data_path=data_path,
                                                    patch_size=patch_size)  # (data_nums, feature_nums)
        print('datas_features_shape:', datas_features.shape)
        data_R_path = data_path[0]  # 实部数据集路径
        data_I_path = data_path[1]  # 虚部数据集路径
        data_R_sets = self.get_data_list(data_path=data_R_path)  # (channels, rows, cols)
        data_I_sets = self.get_data_list(data_path=data_I_path)  # (channels, rows, cols)
        dim = data_R_sets.shape  # 数据维度:(channels, rows, columns)
        print('T dim:', dim)
        sum_contrastive_list = 0  # 对比训练集的数据量
        contrastive_list = []  # 对比训练集数据
        data_detail_list = {}  # 数据信息说明
        num = 0
        # flag = 0
        for i in range(0, dim[1] - patch_size[0]):
            for j in range(0, dim[2] - patch_size[1]):
                # 存储划分T矩阵数据集路径, 路径名为:TRIX.xlsx -> X表示第X个数据块(左上到右下)
                # 结构为:
                #   sheet_RX:表示T矩阵第X个位置的实部(左上到右下)
                #   sheet_IX:表示T矩阵第X个位置的虚部
                target_paths = os.path.join(target_path[0], 'TRI' + str(num) + '.xlsx')
                if not os.path.exists(target_paths):  # 若数据路径已经存在, 说明数据已经生成, 则跳过
                    book = xlsxwriter.Workbook(filename=target_paths)  # 生成指定文件名的excel文件
                    for channel in range(dim[0]):  # 对于每一个通道(实部 的数目(9个))
                        sheet = book.add_worksheet('sheet_R' + str(channel))  # 添加指定sheet名的sheet
                        # 写入指定块大小的数据实部
                        for row in range(patch_size[0]):
                            for column in range(patch_size[1]):
                                sheet.write(row, column, data_R_sets[channel][i + row][j + column])
                    for channel in range(dim[0]):  # 对于每一个通道(虚部 的数目(9个))
                        sheet = book.add_worksheet('sheet_I' + str(channel))
                        # 写入指定块大小的数据虚部
                        for row in range(patch_size[0]):
                            for column in range(patch_size[1]):
                                sheet.write(row, column, data_I_sets[channel][i + row][j + column])
                    book.close()
                # 存储划分每个数据点特征集路径, 路径名为:featuresX.xlsx -> X表示第X个数据点(左上到右下)
                target_paths_feature = os.path.join(target_path[1], 'features' + str(num) + '.xlsx')
                if not os.path.exists(target_paths_feature):  # 若数据路径已经存在, 说明数据已经生成, 则跳过
                    features = datas_features[num]
                    book_F = xlsxwriter.Workbook(filename=target_paths_feature)  # 生成指定文件名的excel文件
                    sheet_F = book_F.add_worksheet('sheet')  # 添加指定sheet名的sheet
                    for index, single_feature in enumerate(features):
                        sheet_F.write(0, index, single_feature)
                    book_F.close()
                contrastive_list.append(target_paths + ';' + target_paths_feature + '\n')
                num += 1
                if num % 1000 == 0:
                    print('generate No.' + str(num) + ' dataset')

        random.shuffle(contrastive_list)  # 打乱对比训练集
        with open(contrastive_list_path, 'a') as f:  # 写入测试集信息
            for contrastive_data in contrastive_list:
                f.write(contrastive_data)

        data_detail_list['data_list_path'] = target_path  # 划分好的数据父目录
        data_detail_list['dim'] = [dim[1] - patch_size[0], dim[2] - patch_size[1]]  # 总数据集大小(rows, cols)
        data_detail_list['contrastive_train_num'] = num  # 训练数据量
        jsons = json.dumps(data_detail_list, sort_keys=True, indent=4, separators=(',', ':'))  # 将字典转化为json形式
        with open(os.path.join(target_path[0].rsplit('/', maxsplit=1)[0], 'readme.json'), 'w') as f:  # 写入数据信息
            f.write(jsons)
        print('生成数据列表完成！')

    def save_contrastive_data_origin(self, data_path, target_path, contrastive_list_path, patch_size):
        """
        保存对比学习训练数据
        readme.json文件保存在二维数据块位置的父目录的父目录中
        :param data_path: T矩阵的实部数据位置和虚部数据位置(list)
        :param target_path: 分割后的数据保存位置(list), 包括二维数据块位置与一维特征集位置
        :param contrastive_list_path: 对比训练集说明文件 --> str
        :param patch_size: 切分的数据块的大小([row, column]) --> tuple
        :return: None
        """
        data_R_path = data_path[0]  # 实部数据集路径
        data_I_path = data_path[1]  # 虚部数据集路径
        data_R_sets = self.get_data_list(data_path=data_R_path)  # (channels, rows, cols)
        data_I_sets = self.get_data_list(data_path=data_I_path)  # (channels, rows, cols)
        dim = data_R_sets.shape  # 数据维度:(channels, rows, columns)
        sum_contrastive_list = 0  # 对比训练集的数据量
        contrastive_list = []  # 对比训练集数据
        data_detail_list = {}  # 数据信息说明
        # num = 0
        # flag = 0
        for i in range(0, dim[1] - patch_size[0]):
            for j in range(0, dim[2] - patch_size[1]):
                T = np.zeros((3, 3), dtype='complex')
                # 存储划分T矩阵数据集路径, 路径名为:TRIX.xlsx -> X表示第X个数据块(左上到右下)
                # 结构为:
                #   sheet_RX:表示T矩阵第X个位置的实部(左上到右下)
                #   sheet_IX:表示T矩阵第X个位置的虚部
                target_paths = os.path.join(target_path[0], 'TRI' + str(sum_contrastive_list) + '.xlsx')
                if not os.path.exists(target_paths):  # 若数据路径已经存在, 说明数据已经生成, 则跳过
                    book = xlsxwriter.Workbook(filename=target_paths)  # 生成指定文件名的excel文件
                    for channel in range(dim[0]):  # 对于每一个通道(实部 的数目(9个))
                        sheet_R = book.add_worksheet('sheet_R' + str(channel))  # 添加指定sheet名的sheet
                        sheet_I = book.add_worksheet('sheet_I' + str(channel))
                        # 写入指定块大小的数据实部
                        for row in range(patch_size[0]):
                            for column in range(patch_size[1]):
                                sheet_R.write(row, column, data_R_sets[channel][i + row][j + column])
                                sheet_I.write(row, column, data_I_sets[channel][i + row][j + column])
                        T[channel // 3, channel % 3] = data_R_sets[channel][i + patch_size[0] // 2][
                                                           j + patch_size[1] // 2] + \
                                                       data_I_sets[channel][i + patch_size[0] // 2][
                                                           j + patch_size[1] // 2] * 1j
                    book.close()
                # 存储划分每个数据点特征集路径, 路径名为:featuresX.xlsx -> X表示第X个数据点(左上到右下)
                target_paths_feature = os.path.join(target_path[1], 'features' + str(sum_contrastive_list) + '.xlsx')
                if not os.path.exists(target_paths_feature):  # 若数据路径已经存在, 说明数据已经生成, 则跳过
                    features = self.get_feature_by_T_metric(data=T)
                    book_F = xlsxwriter.Workbook(filename=target_paths_feature)  # 生成指定文件名的excel文件
                    sheet_F = book_F.add_worksheet('sheet')  # 添加指定sheet名的sheet
                    for index, single_feature in enumerate(features):
                        sheet_F.write(0, index, single_feature)
                    book_F.close()
                if random.random() < 0.1:
                    contrastive_list.append(target_paths + ';' + target_paths_feature + '\n')
                    sum_contrastive_list += 1
                    if sum_contrastive_list % 1000 == 0:
                        print('generate No.' + str(sum_contrastive_list) + ' dataset')

        random.shuffle(contrastive_list)  # 打乱对比训练集
        with open(contrastive_list_path, 'a') as f:  # 写入测试集信息
            for contrastive_data in contrastive_list:
                f.write(contrastive_data)

        data_detail_list['data_list_path'] = target_path  # 划分好的数据父目录
        data_detail_list['dim'] = [dim[1] - patch_size[0], dim[2] - patch_size[1]]  # 总数据集大小(rows, cols)
        data_detail_list['contrastive_train_num'] = sum_contrastive_list  # 训练数据量
        jsons = json.dumps(data_detail_list, sort_keys=True, indent=4, separators=(',', ':'))  # 将字典转化为json形式
        with open(os.path.join(target_path[0].rsplit('/', maxsplit=1)[0], 'readme.json'), 'w') as f:  # 写入数据信息
            f.write(jsons)
        print('generate contrastive dataset succeed！')

    def data_dim_change(self, data):
        """
        数据的维度变换:将(channels, rows, cols)转换为(rows, cols, channels)
        :param data: 原数据
        :return: 转换后的数据
        """
        dim = data.shape  # 原数据维度(channels, rows, cols)
        new_data = np.zeros((dim[1], dim[2], dim[0]))  # 转换后数据(rows, cols, channels)
        for channel in range(dim[0]):
            for row in range(dim[1]):
                for column in range(dim[2]):
                    new_data[row][column][channel] = data[channel][row][column]  # 维度转换
        return new_data
