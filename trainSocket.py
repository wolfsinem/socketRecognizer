# Convert Indentation to Spaces --> use for error
import os
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model

dataset_dir = '/Users/wolfsinem/Downloads/socketdata/'
category = 'dataset'

class SocketDataset(mrcnn.utils.Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class(category, 1, "AOP_BTV1")
        self.add_class(category, 2, "AOP_DIO_01")
        self.add_class(category, 3, "AOP_EVK80")
        self.add_class(category, 4, "AOP_TRAS1000")
        self.add_class(category, 5, "AOP_TRAS1000_no_key")
        self.add_class(category, 6, "AOP_X10DER_KT_01")
        self.add_class(category, 7, "SPLITTER_MCP_03")
        self.add_class(category, 8, "SPLITTER_POA_01IEC")
        self.add_class(category, 9, "SPLITTER_POA_01_met_kapje")
        self.add_class(category, 10, "SPLITTER_POA_01_zonder_kapje")
        self.add_class(category, 11, "SPLITTER_POA_3_met_kapje")
        self.add_class(category, 12, "SPLITTER_POA_3_zonder_kapje")
        self.add_class(category, 13, "SPLITTER_SQ601_met_kapje")
        self.add_class(category, 14, "SPLITTER_UMU_met_kapje")
        self.add_class(category, 15, "WCD_tweegats")

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

            self.add_image(category, image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for object in root.findall('.//object'):
            box_class_list = list()
            for box in root.findall('.//bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                coors = [xmin, ymin, xmax, ymax]
                box_class_list.append(coors)
                for i in box_class_list:
                    box_class_list = [[int(x) for x in i]]

            for name in object.findall('.//name'):
                box_class_list.append(name.text)

            boxes.append(box_class_list)

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
            box = boxes[i][0]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(boxes[i][1]))
        
        return masks, asarray(class_ids, dtype='int32')


class SocketConfig(mrcnn.config.Config):
    NAME = "socketConfig"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 15
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