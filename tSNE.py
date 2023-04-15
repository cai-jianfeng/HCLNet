# In[]
import random

import numpy as np

from sklearn.manifold import TSNE

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import torch

from data_preprocess import Data
from network import DClass_Net
from tqdm import tqdm

patch_size = [15, 15]
data_path = ['/path/to/T_R.xlsx', '/path/to/T_I.xlsx']
label_path = '/path/to/label.xlsx'
num_class = 0
data = Data()
data_R_set = data.get_data_list(data_path=data_path[0])
data_T_set = data.get_data_list(data_path=data_path[1])
label_set = data.get_label_list(label_path=label_path)
dim = data_R_set.shape
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
cnet = DClass_Net()
cnet_param = torch.load('/path/to/xxx.pkl', map_location=device)
cnet.load_state_dict(cnet_param)

# In[]
rs = tqdm(range(0, dim[1] - patch_size[0]))

representations = []

y = []
with torch.no_grad():
    for i in rs:
        for j in range(0, dim[2] - patch_size[1]):
            label = int(label_set[i + patch_size[0] // 2][j + patch_size[1] // 2])
            if random.random() < 0.1 and label != 0:
                data_R = data_R_set[:, i:i + patch_size[0], j:j + patch_size[1]]
                data_T = data_T_set[:, i:i + patch_size[0], j:j + patch_size[1]]
                predict_data = np.concatenate((data_R, data_T), axis=0)
                predict_data = np.expand_dims(predict_data, 0)
                predict_data = torch.tensor(predict_data, dtype=torch.float32)
                predict_data = predict_data.to(device)
                representation = cnet(predict_data).squeeze(0)
                representations.append(representation.cpu().detach().numpy())
                y.append(label)
            rs.desc = 'predict -> {}/{}'.format(i, j)

# In[]
def plot(x, colors, pig_name, num_class):
    # Choosing color palette
    palette = np.array(sns.color_palette("husl", num_class+1))
    # pastel, husl, and so on

    # Create a scatter plot.
    f = plt.figure(figsize=(16, 16))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], c=palette[colors.astype(np.int8)])
    # Add the labels for each digit.
    txts = []
    # for i in range(2):
    for i in range(num_class+1):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    plt.savefig(pig_name, dpi=120)
    return f, ax, txts


# Place the arrays of data of each digit on top of each other and store in X
X = np.array(representations)
Y = np.array(y)
# Implementing the TSNE Function - ah Scikit learn makes it so easy!
TX = TSNE(perplexity=50).fit_transform(X)
plot(TX, Y, '/path/to/tsne.png', num_class=num_class)
