"""
created on:2022/11/18 15:16
@author:caijianfeng
@Purpose: this code file covers
            1. CL, fine-tuning/train, eval and predict step;
            2. result calculate(OA, AA, Kappa)
"""
import warnings

import torch
import os
import sys
import numpy as np
import math
import xlsxwriter
import xlrd
from tqdm import tqdm


class run:
    def __init__(self):
        pass

    def contrastive(self, model, train_loader, optimizer, criterion, cost, device=None):
        # running_loss = 0.0
        # contrastive_bar = tqdm(train_loader, file=sys.stdout)
        for batch_idx, data in enumerate(train_loader):
            data_T, data_F = data
            data_T, data_F = data_T.to(device), data_F.to(device)
            optimizer.zero_grad()
            logits, labels = model(data_F, data_T)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            # contrastive_bar.desc = 'train epoch[{}/{}] loss:{: .3f}'.format(epoch[0] + 1, epoch[1], running_loss)
            cost.append(running_loss)
        cost.pop()

    def train(self, model, epoch, train_loader, optimizer, criterion, cost, device):
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for batch_idx, data in enumerate(train_bar):
            inputs, label = data
            inputs, label = inputs.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label - 1)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            train_bar.desc = 'train epoch[{}/{}] loss:{: .3f}'.format(epoch[0] + 1, epoch[1], running_loss)
            cost.append(running_loss)

    def eval(self, model, epoch, eval_loader, device, accuracy):
        correct = 0
        total = 0
        with torch.no_grad():
            eval_bar = tqdm(eval_loader, file=sys.stdout)
            for id, data in enumerate(eval_bar):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # print('img_dtype:', type(images))
                # print('img_shape:', images.shape)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                # print('predict_shape:', predicted.shape, 'label_shape:', labels.shape)
                total += labels.size(0)
                correct += (predicted == (labels - 1)).sum().item()
                acc = 100 * correct / total
                eval_bar.desc = 'eval epoch[{}/{}] Accuracy{: .3f}%'.format(epoch[0] + 1, epoch[1], acc)
            accuracy.append(acc)

    def predict(self, model, predict_datas, target_path, color_path, label_pic_name, transform, device, name):
        rows = predict_datas.shape[0]
        columns = predict_datas.shape[1]
        # predict_labels = [[0 for _ in range(columns)] for _ in range(rows)]
        target_path = os.path.join(target_path, 'predict_labels' + name + '.xlsx')
        book = xlsxwriter.Workbook(filename=target_path)
        sheet = book.add_worksheet('sheet')
        rs = tqdm(range(rows))
        for row in range(rs):
            # cols = tqdm(range(columns))
            for column in range(columns):
                predict_data = predict_datas[row][column]
                # predict_data = transform(predict_data)
                predict_data = np.expand_dims(predict_data, 0)
                predict_data = torch.tensor(predict_data, dtype=torch.float32)
                predict_data = predict_data.to(device)
                # print('predict_data_dtype:', type(predict_data))
                # print('predict_data_shape:', predict_data.shape)
                predict_label = model(predict_data)
                _, predict_label = torch.max(predict_label.data, dim=1)
                # print('predict_label:', predict_label)
                # predict_labels[row][column] = predict_label
                # print('predict_label:', predict_label.item())
                sheet.write(row, column, predict_label.item())
            rs.desc = 'predict -> row: {}/{}'.format(row, rows)
        book.close()
        # plot_mod = plot_mode(mode='predict')
        # plot_mod.plot_labels(label_path=target_path,
        #                      color_path=color_path,
        #                      label_pic_name=label_pic_name)

    def adjust_learning_rate(self, optimizer, epoch, epochs, lr):
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def overall_accuracy(self, predict_path, label_path, patch_size=None):
        """
        calculate predicted accuracy
        :param patch_size: traing data_flevoland4 patch size -> list
        :param predict_path: predicted labels file path -> str
        :param label_path: ground truth file path -> str
        :return: accuracy -> float
        """
        if patch_size is None:
            patch_size = [15, 15]
        predict_book = xlrd.open_workbook(filename=predict_path)
        predict_sheet = predict_book.sheet_by_index(0)
        label_book = xlrd.open_workbook(filename=label_path)
        label_sheet = label_book.sheet_by_index(0)
        rows = predict_sheet.nrows
        cols = predict_sheet.ncols
        acc = 0
        num = 0
        for row in range(rows):
            for col in range(cols):
                predict = int(predict_sheet.cell_value(row, col))
                label = int(label_sheet.cell_value(row + patch_size[0] // 2, col + patch_size[1] // 2))
                acc += 1 if label != 0 and predict == (label - 1) else 0
                num += 1 if label != 0 else 0
        acc /= num
        return acc

    def average_accuracy(self, predict_path, label_path, patch_size, num_class=4):
        """
        calculate predicted accuracy
        :param num_class: dataset categories
        :param patch_size: traing data_flevoland4 patch size -> list
        :param predict_path: predicted labels file path -> str
        :param label_path: ground truth file path -> str
        :return: accuracy -> float
        """
        predict_book = xlrd.open_workbook(filename=predict_path)
        predict_sheet = predict_book.sheet_by_index(0)
        label_book = xlrd.open_workbook(filename=label_path)
        label_sheet = label_book.sheet_by_index(0)
        rows = predict_sheet.nrows
        cols = predict_sheet.ncols
        accs = [0 for _ in range(num_class)]
        nums = [0 for _ in range(num_class)]
        for row in range(rows):
            for col in range(cols):
                predict = int(predict_sheet.cell_value(row, col))
                label = int(label_sheet.cell_value(row + patch_size[0] // 2, col + patch_size[1] // 2))
                if label != 0:
                    label -= 1
                    accs[label] += 1 if predict == label else 0
                    nums[label] += 1
        for i in range(num_class):
            accs[i] /= nums[i]
        AA = sum(accs) / len(accs)
        return [accs, AA]

    def Kappa(self, predict_path, label_path, patch_size, num_class):
        """
        calculate predicted accuracy
        :param patch_size: traing data_flevoland4 patch size -> list
        :param predict_path: predicted labels file path -> str
        :param label_path: ground truth file path -> str
        :return: accuracy -> float
        """
        predict_book = xlrd.open_workbook(filename=predict_path)
        predict_sheet = predict_book.sheet_by_index(0)
        label_book = xlrd.open_workbook(filename=label_path)
        label_sheet = label_book.sheet_by_index(0)
        rows = predict_sheet.nrows
        cols = predict_sheet.ncols
        accs = [0 for _ in range(num_class)]
        pred = [0 for _ in range(num_class)]
        nums = [0 for _ in range(num_class)]
        num = 0
        kappa = 0
        for row in range(rows):
            for col in range(cols):
                predict = int(predict_sheet.cell_value(row, col))
                label = int(label_sheet.cell_value(row + patch_size[0] // 2, col + patch_size[1] // 2))
                if label != 0:
                    label -= 1
                    accs[label] += 1 if predict == label else 0  # A
                    nums[label] += 1  # A + C
                    pred[predict] += 1  # A + B
                    num += 1
        for i in range(num_class):
            A = accs[i]
            B = pred[i] - A
            C = nums[i] - A
            D = num - A - B - C
            R1 = A + B
            R2 = C + D
            C1 = A + C
            C2 = B + D
            kappa += (num * (A + D) - (R1 * C1 + R2 * C2)) / (num * num - (R1 * C1 + R2 * C2))
        kappa /= num_class
        return kappa

    def result_smooth(self, result_path):
        """
        warning: this code function if cancellation
        (predict) result smooth
        :param result_path: predict result label path -> str
        :return: smooth labels -> numpy.array
        """
        warnings.warn("this code function if cancellation", DeprecationWarning)
        super_patch = 15
        num_class = 0
        result_book = xlrd.open_workbook(result_path)
        result_sheet = result_book.sheet_by_index(0)
        rows = result_sheet.nrows
        cols = result_sheet.ncols
        result = np.zeros((rows, cols))

        for row in range(0, rows, super_patch):
            for col in range(0, cols, super_patch):
                most = np.zeros(num_class)
                if row < rows - super_patch and col < cols - super_patch:
                    for i in range(super_patch):
                        for j in range(super_patch):
                            most[int(result_sheet.cell_value(row + i, col + j))] += 1
                    most_idx = np.argmax(most)
                    for i in range(super_patch):
                        for j in range(super_patch):
                            result[row + i][col + j] = most_idx
        return result
