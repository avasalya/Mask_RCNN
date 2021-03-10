# %% [markdown]
# # Mask R-CNN - Train on Satellite Shapes Dataset
#
#
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (Housess, Sheds/Garages, and Buildings) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
#
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster.

# %%
# Configurations
import os
import re
import sys
import cv2
import time
import math
import random
import random as rand
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# data directory path
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RGB_DIR = os.path.join(DATA_DIR, 'raw')

# define classes
class_names = ['BG', 'Houses', 'Buildings', 'Sheds/Garages']

# total samples available
totalSamples = os.listdir(RGB_DIR)

# create training and testing sets
train_idx, valid_idx = train_test_split(totalSamples, test_size = 0.20)
eval_idx = os.listdir(os.path.join(DATA_DIR, 'eval'))

# "Return a Matplotlib Axes array to be used in all visualizations in the notebook
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class ShapesConfig(Config):
    """Configuration for training on the satellite dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = ShapesConfig()
# config.display()

# %% [markdown]

class SatelliteDataset(utils.Dataset):
    """The dataset consists of three classes:
        (Houses, Buildings, Sheds/Garages)
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "Houses")
        self.add_class("shapes", 2, "Buildings")
        self.add_class("shapes", 3, "Sheds/Garages")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                        width=width, height=height,
                        bg_color=bg_color, shapes=shapes)


    def random_image(self, height, width):

        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])

        samples, boxes, areas, shapes = self.load_sample(count=2,
                                                        dataset=dataset,
                                                        scaled=True,
                                                        scaled_to=30,
                                                        show_fig=True)

        # Apply non-max suppression wit 0.3 threshold to avoid shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(len(boxes)), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]

        return bg_color, shapes#('classname', 'color', 'bbox')

    def load_image(self, samples, dataset=train_idx):
        """This function loads the image from a file using,
        same random sample as generated using method `random_image`
        """
        for sample in samples:
            frame_id = dataset[sample]
            imagePath = os.path.join(RGB_DIR, frame_id)
            fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
            rgb = plt.imread(imagePath)
            plt.imshow(rgb)
            plt.show()


    def load_sample(self, count=1, dataset=train_idx, scaled=False, scaled_to=50, show_fig=True):
        """load the requested number of images.
        count: number of images to generate.
        scaled: whether to resize image or not.
        scaled_to: percentage to resize the image.
        """
        # choose random sample(s)
        samples = rand.sample(range(0, len(dataset)), count)

        # MAIN Loop
        for image_id, sample in enumerate(samples):

            # resize images
            frame_id = dataset[sample]
            imagePath = os.path.join(RGB_DIR, frame_id)
            self.image, self.width, self.height = self.scale_image(plt.imread(imagePath),
                                                                    scaled=scaled,
                                                                    scaled_to=scaled_to)

            # record polygons class their bounding boxes and areas
            shapes = []
            boxes = []
            areas = []
            list_vertices = []

            # read polygon annotations
            data = pd.read_json(imagePath.replace('raw', 'annotations').replace('png', 'png-annotated.json'))

            for shape in range(len(data.labels)):
                print('found {} {}'.format(len(data.labels[shape]['annotations']), data.labels[shape]['name']))

                # iterate thorough each polygons
                for poly in range(len(data.labels[shape]['annotations'])):

                    # get vertices of polygons (house, building, garage)
                    vertices = np.array(data.labels[shape]['annotations'][poly]['segmentation'], np.int32)
                    vertices = vertices.reshape((-1,1,2))

                    # draw polygons on scaled image
                    if scaled == True:
                        scaled_vertices = []
                        for v in range(len(vertices)):
                            scaled_vertices.append(int(vertices[v][0][0] * scaled_to / 100)) #x
                            scaled_vertices.append(int(vertices[v][0][1] * scaled_to / 100)) #y
                        vertices = np.array(scaled_vertices).reshape((-1,1,2))

                    # draw polygons on scaled image to create segmentation
                    image, color, bbox, area = self.draw_polygons(self.image,
                                                                vertices,
                                                                shape,
                                                                draw_bbox=False)

                    # same length as total polygons
                    boxes.append(bbox)
                    areas.append(area)
                    shapes.append((data.labels[shape]['name'], color, bbox))
                    list_vertices.append(vertices)

            # create mask for each instances
            mask, class_ids = self.load_mask(self.width, self.height, shapes, list_vertices, len(boxes))

            # # Apply non-max suppression wit 0.3 threshold to avoid shapes covering each other
            # keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(len(boxes)), 0.3)
            # shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]

            if show_fig == True:
                fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
                plt.imshow((image* 255).astype(np.uint8))
                plt.show()
                visualize.display_top_masks(self.image, mask, class_ids, class_names)

        return samples, boxes, areas, shapes

    def scale_image(self, image, scaled=False, scaled_to=50):
        """scale original image to speed-up training
        """
        if scaled == True:
            #calculate the 50 percent of original dimensions
            width = int(image.shape[1] * scaled_to / 100)
            height = int(image.shape[0] * scaled_to / 100)

            # if input image is too big, resize it
            dsize = (width, height)
            new_image = cv2.resize(image, dsize, interpolation = cv2.INTER_CUBIC)
            print("resized image from {} to {}".format(image.shape, new_image.shape))
        else:
            new_image = image

        return new_image, new_image.shape[0], new_image.shape[1]

    def draw_polygons(self, image, vertices, shape, draw_bbox=True):

        color = tuple([random.randint(0, 255) for _ in range(3)])
        # print(color)

        # draw segmented polygon
        cv2.drawContours(image, [vertices], contourIdx= 0, color= color, thickness= -1)

        # compute the bounding boxes from instance masks
        rect = cv2.minAreaRect(vertices)
        bbox = cv2.boxPoints(rect)
        bbox = np.int0(bbox)

        # coordinates of bounding box
        top_left = np.min(bbox, axis=0)#x1,y1
        bottom_right = np.max(bbox, axis=0)#,x2,y2

        # plot bounding box
        if draw_bbox:
            cv2.drawContours(image, [bbox] , 0, color, 4)
            # print(bbox)

        # get area of bounding box
        area = cv2.contourArea(vertices)

        return image, color, np.append(top_left, bottom_right), area

    def load_mask(self, width, height, shapes, list_vertices, total_instances):

        """Generate instance masks for shapes of the given image ID.
        """
        # create empty mask
        mask = np.zeros([ width, height, total_instances], dtype=np.uint8)

        # fill each channel with an annotated polygon
        for i, (shape, _, dims, vertices) in enumerate(shapes, list_vertices):
            mask[...,i:i+1] = cv2.drawContours(mask[...,i:i+1].copy(), [vertices], contourIdx= 0, color=1, thickness= -1)

        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(total_instances-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # Map class names to class IDs.
        class_ids = np.array([class_names.index(s[0]) for s in shapes])

        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)



dataset_train = SatelliteDataset()



train_samples, train_boxes, train_areas, train_shapes = dataset_train.load_sample(count=1,
                                                        dataset=train_idx,
                                                        scaled=True,
                                                        scaled_to=30,
                                                        show_fig=True)


dataset_train.load_shapes(count=500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()


# # Training dataset
# train_samples, train_boxes, train_areas, train_shapes = dataset_train.random_image(count=2,
#                                                         dataset=train_idx,
#                                                         scaled=True,
#                                                         scaled_to=30,
#                                                         show_fig=True)
# dataset_train.prepare()
# # dataset_train.load_image(train_samples, train_idx)

# # Validation dataset
# dataset_val = SatelliteDataset()
# val_samples, val_boxes, val_areas, val_shapes = dataset_val.random_image(count=2,
#                                                         dataset=valid_idx,
#                                                         scaled=True,
#                                                         scaled_to=30,
#                                                         show_fig=False)
# dataset_val.prepare()

#%%

print(dataset_train)

# %% [markdown]
# ## Create Model

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                        model_dir=MODEL_DIR)


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# %% [markdown]
# ## Training
#
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
#
# 2. Fine-tune all layers. Simply pass `layers="all` to train all layers.

# %%
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')


# %%
# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")


# %%
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

# %% [markdown]
# ## Detection

# %%
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# %%
# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))


# %%
results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

# %% [markdown]
# ## Evaluation

# %%
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))


# %%



