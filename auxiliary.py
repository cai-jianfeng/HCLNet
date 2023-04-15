# _*_ coding:utf-8 _*_
"""
@Software: code
@FileName: result_show.py
@Date: 2023/3/5 11:44
@Author: caijianfeng
@Purpose:
    1. concat two Excel file: In the case of prediction, it may not be possible to predict directly at once,
    and it is necessary to predict several times. In this case, it is necessary to combine the results(Excel files) of several times
    2. plot label map: plot the whole category map(predict/ground-truth) using your color scheme
    3. plot mask label map: plot the whole category map(predict) using your color scheme,
    when the ground-truth of the pixel is None, it will be masked and not plot(plot as black)
"""

import xlrd
import xlsxwriter
from tqdm import tqdm
from plot import plot_mode

# In[1] concat two Excel(.xlsx) file into a new Excel file (each Excel file only have 1 sheet)
excel_file1 = '/path/to/xxx.xlsx'  # the first Excel file name you want to concat
excel_file2 = '/path/to/xxx.xlsx'  # the second Excel file name you want to concat

# open Excel file
predict_book1 = xlrd.open_workbook(excel_file1)
predict_book2 = xlrd.open_workbook(excel_file2)

save_file = '/path/to/xxx.xlsx'  # the saving name after concat
# using xlsxwriter to write the concat result
predict_book = xlsxwriter.Workbook(filename=save_file)
predict_sheet1 = predict_book1.sheet_by_index(0)
predict_sheet2 = predict_book2.sheet_by_index(0)
rows1 = predict_sheet1.nrows
rows2 = predict_sheet2.nrows
cols = predict_sheet1.ncols  # the rows in both Excel files can be different, but the cols must be same
# concat data will save in the first sheet
predict_sheet = predict_book.add_worksheet(name='0')

rows = tqdm(range(rows1))
for row in rows:
    for col in range(cols):
        predict_sheet.write(row, col, predict_sheet1.cell_value(row, col))
        rows.desc = 'row: {} / {}'.format(row, rows1 + rows2)

rows = tqdm(range(rows2))
for row in rows:
    for col in range(cols):
        predict_sheet.write(row + rows1, col, predict_sheet2.cell_value(row, col))
        rows.desc = 'row: {} / {}'.format(row + rows1, rows1 + rows2)
predict_book.close()
print('predict label concat succeed!')

# In[2] plot predict/ground-truth label map
target_path = '/path/to/xxx.xlsx'  # the predicted/ground-truth label
color_path = '/path/to/xxx.xlsx'  # the colors corresponding to different categories (RGB: 3 columns x N categories)
label_pic_name = '$map name$'  # the save file name of ploted map
plot_mod = plot_mode(mode='predict')
plot_mod.plot_labels(label_path=target_path,
                     color_path=color_path,
                     label_pic_name=label_pic_name)
print('create picture succeed!')

# In[3] plot mask predict label
label_path = '/path/to/label.xlsx'  # the predicted label
predict_path = '/path/to/predict.xlsx'  # the ground-truth label
color_path = '/path/to/color.xlsx'  # the colors corresponding to different categories (RGB: 3 columns x N categories)
label_pic_name_mask = '$mask map name$'  # the save file name of ploted map
patch_size = []  # the patch size you used for training (in this article is [15, 15])
plot_mod = plot_mode(mode='predict')
plot_mod.plot_mask_labels(label_path=label_path,
                          predict_path=predict_path,
                          color_path=color_path,
                          label_pic_name=label_pic_name_mask,
                          patch_size=patch_size)
print('create mask picture succeed!')

