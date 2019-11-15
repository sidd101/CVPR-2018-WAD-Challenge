from mrcnn import visualize
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from celluloid import Camera
import math
import random
import numpy as np

class AnimationConfig:
  def __init__(self):
    self.fig = plt.figure(figsize=(10, 10))
    self.ax = plt.Axes(self.fig, [0, 0, 1, 1])
    self.ax.set_axis_off()
    self.fig.add_axes(self.ax)
    self.camera = Camera(self.fig)
  def get_camera(self):
    return self.camera
  def get_ax(self):
    return self.ax
  
########################################################################################################################
#  snapping each image along with bbox, mask and class name. 
#  This methods uses reference of AnimationConfig class as parameter and using celluloid camera for snaping each image. 
#  
#  Note: This is a clone method from matterport maskRCNN utils.py with small tweaks to creation animation from sequence of
#  images
########################################################################################################################
def snap_instance(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",animation=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    if not animation:
         print("\n*** No animation object passed for snapping *** \n")
         return

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    animation.get_ax().set_ylim(height + 10, -10)
    animation.get_ax().set_xlim(-10, width + 10)
    animation.get_ax().axis('off')
    animation.get_ax().set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            animation.get_ax().add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        animation.get_ax().text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            animation.get_ax().add_patch(p)
    animation.get_ax().imshow(masked_image.astype(np.uint8))
    animation.get_camera().snap()
