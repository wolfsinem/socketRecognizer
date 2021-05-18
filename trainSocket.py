# Convert Indentation to Spaces --> use for error
import os
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model

dataset_dir = '/Users/wolfsinem/Downloads/socketdata/'

class SocketDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "socket")

        images_dir = dataset_dir + '/Images/'
        annotations_dir = dataset_dir + '/Annots/'

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            if is_train and int(image_id) >= 150:
                continue

            if not is_train and int(image_id) < 150:
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        
        for i in boxes:
            boxes = [[int(x) for x in i]]


        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('socket'))
        return masks, asarray(class_ids, dtype='int32')

class SocketConfig(mrcnn.config.Config):
    NAME = "socketConfig"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 100

train_set = SocketDataset()
train_set.load_dataset(dataset_dir=dataset_dir, is_train=True)
train_set.prepare()

valid_dataset = SocketDataset()
valid_dataset.load_dataset(dataset_dir=dataset_dir, is_train=False)
valid_dataset.prepare()

socket_config = SocketConfig()

model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=socket_config)

model.load_weights(filepath='/Users/wolfsinem/maskrcnn/mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_set, 
            val_dataset=valid_dataset, 
            learning_rate=socket_config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

model_path = 'socket_mask_rcnn.h5'
model.keras_model.save_weights(model_path)