import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
img_file= load_sample_image("china.jpg")
ax=plt.axes(xticks=[], yticks=[])
ax.imshow(img_file)

img_file.shape
img_file=img_file/255
data=img_file.reshape(427*640,3)

from sklearn.cluster import MiniBatchKMeans
kmns=MiniBatchKMeans(16).fit(data)
kcolors=kmns.cluster_centers_[kmns.predict(data)]

import numpy as np
kimg=np.reshape(kcolors,(img_file.shape))

fig,(axes1,axes2)=plt.subplots(1,2)
fig.suptitle('kmnns image compressor',fontsize=20)
axes1.set_title('compressed image')
axes1.set_xticks([])
axes1.set_yticks([])
axes1.imshow(kimg)

axes2.set_title('original image')
axes2.set_xticks([])
axes2.set_yticks([])
axes2.imshow(kimg)
plt.subplots_adjust(top=0.85)
plt.show()

import matplotlib
matplotlib.image.imsave('compressed.jpg',kimg)
