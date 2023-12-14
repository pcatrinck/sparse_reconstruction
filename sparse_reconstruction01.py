import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy import ndimage

def plane_sweep_ncc(im_l, im_r, start, steps, wid):
    """ Find disparity image using normalized cross-correlation. """
    m, n = im_l.shape

    # arrays to hold the different sums
    mean_l = np.zeros((m, n))
    mean_r = np.zeros((m, n))
    s = np.zeros((m, n))
    s_l = np.zeros((m, n))
    s_r = np.zeros((m, n))
    # array to hold depth planes
    dmaps = np.zeros((m, n, steps))

    # compute mean of patch
    ndimage.uniform_filter(im_l, wid, mean_l)
    ndimage.uniform_filter(im_r, wid, mean_r)

    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r

    # try different disparities
    for displ in range(steps):
        # move left image to the right, compute sums

        ndimage.uniform_filter(norm_l * np.roll(norm_r, displ + start), wid, s)  # sum nominator
        ndimage.uniform_filter(norm_l * norm_l, wid, s_l)
        ndimage.uniform_filter(
            np.roll(norm_r, displ + start) * np.roll(norm_r, displ + start), wid, s_r
        )  # sum denominator
        # store ncc scores
        dmaps[:, :, displ] = s / np.sqrt(np.absolute(s_l * s_r))

    # pick best depth for each pixel
    best_map = np.argmax(dmaps, axis=2) + start

    return best_map


# intrinsic parameter matrix
fm = 403.657593  # Focal distantce in pixels
cx = 161.644318  # Principal point - x-coordinate (pixels)
cy = 124.202080  # Principal point - y-coordinate (pixels)
bl = 119.929  # baseline (mm)

# taking images
im_l = np.array(Image.open('imgs\esquerda.ppm').convert('L'), 'f')
im_r = np.array(Image.open('imgs\direita.ppm').convert('L'), 'f')

# starting displacement and steps
steps = 44
start = 4
m, n = im_l.shape
wid1 = 11 # width for ncc

disp_map = plane_sweep_ncc(im_l, im_r, start, steps, wid1)

# Calculate depths
z3d = np.ones(disp_map.shape) * fm * bl
z3d = np.divide(z3d, disp_map)

u, v = np.meshgrid(np.arange(n), np.arange(m))

x3d = np.multiply(((u - cx) / fm), z3d)
y3d = np.multiply(((v - cy) / fm), z3d)

# Filter erroneous depths
good = np.where((z3d > 0) & (z3d < 2500))

x3d = x3d[good]
y3d = y3d[good]
z3d = z3d[good]

u = u[good]
v = v[good]

# Take the original image colors
pixel_color = []
im_l = cv2.imread('imgs\esquerda.ppm')
print(im_l.shape)

for i in range(u.shape[0]):
    pixel_color.append(im_l[int(v[i]), int(u[i])])

pixel_color = np.asarray(pixel_color)

# Plot 3D points with their original colors from the image
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x3d, y3d, z3d, c=pixel_color/255.0)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Set view_init for the first figure
ax.view_init(elev=-85, azim=-100)

# Create the second figure
fig2 = plt.figure(figsize=(10,10))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(x3d, y3d, z3d, c=pixel_color/255.0)
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')

# Set view_init for the second figure
ax2.view_init(elev=-23,azim=-91)

plt.show()