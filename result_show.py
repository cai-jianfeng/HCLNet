# _*_ coding:utf-8 _*_
"""
@Software: code
@FileName: result_show.py
@Date: 2022/12/5 11:40
@Author: caijianfeng
"""
from net_run import run

# In[1] calculate OA
print('calculate OA/AA/Kappa begin!')
run_util = run()
predict_path = '/path/to/predict.xlsx'
label_path = '/path/to/label.xlsx'
patch_size = []
num_class = 0
OA = run_util.overall_accuracy(predict_path=predict_path,
                               label_path=label_path,
                               patch_size=patch_size)
print(OA)

# In[2] calculate AA
accs, AA = run_util.average_accuracy(predict_path=predict_path,
                                     label_path=label_path,
                                     patch_size=patch_size,
                                     num_class=num_class)
print(AA)

# In[3] calculate Kappa

kappa = run_util.Kappa(predict_path=predict_path,
                       label_path=label_path,
                       patch_size=patch_size,
                       num_class=num_class)
print('OA:', OA, '; AA:', AA, '; kappa:', kappa)

for i, acc in enumerate(accs):
    print('class {} acc is: {}'.format(i+1, acc))