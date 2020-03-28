import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

plt.rcParams["figure.figsize"]= (10,10)
np.set_printoptions(precision=3)

triangle = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype='float')

mask = triangle
kernel = cv2.getGaussianKernel(11, 5.)
kernel = kernel * kernel.transpose() * mask # Is the 2D filter
kernel = kernel / np.sum(kernel)
print(kernel)

image = cv2.imread('person.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

def bokeh(image):
    r,g,b = cv2.split(image)

    r = r / 255.
    g = g / 255.
    b = b / 255.

    r = np.where(r > 0.9, r * 2, r)
    g = np.where(g > 0.9, g * 2, g)
    b = np.where(b > 0.9, b * 2, b)

    fr = cv2.filter2D(r, -1, kernel)
    fg = cv2.filter2D(g, -1, kernel)
    fb = cv2.filter2D(b, -1, kernel)

    fr = np.where(fr > 1., 1., fr)
    fg = np.where(fg > 1., 1., fg)
    fb = np.where(fb > 1., 1., fb)

    result = cv2.merge((fr, fg, fb))
    return result

result = bokeh(image)
plt.imshow(result)

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = cv2.imread('person.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV uses BGR by default

image = image / 255. # Normalize image
channels_first = np.moveaxis(image, 2, 0) # Channels first

# The pre-trained model expects a float32 type
channels_first = torch.from_numpy(channels_first).float()

prediction = model([channels_first])[0]
scores = prediction['scores'].detach().numpy()
masks = prediction['masks'].detach().numpy()
mask = masks[0][0]
plt.imshow(masks[0][0])

inverted = np.abs(1. - mask)

r,g,b = cv2.split(image)
mr = r * mask
mg = g * mask
mb = b * mask
subject = cv2.merge((mr, mg, mb))

ir = r * inverted
ig = g * inverted
ib = b * inverted
background = cv2.merge((ir, ig, ib))

subject = np.asarray(subject * 255., dtype='uint8')
plt.imshow(subject)

background_bokeh = bokeh(np.asarray(background * 255, dtype='uint8'))
background_bokeh = np.asarray(background_bokeh * 255, dtype='uint8')
combined = cv2.addWeighted(subject, 1., background_bokeh, 1., 0)
plt.imshow(combined)
plt.show()

