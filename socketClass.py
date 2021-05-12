from matplotlib.pyplot import box
import numpy
from mrcnn.utils import Dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
import numpy as np

category = 'socket'

class SocketDataset(Dataset):
    """
    Class that defines and loads license plate dataset
    """

    def load_dataset(self, dataset_dir, is_train=True, train_max=81):
        """[summary]

        Args:
            dataset_dir ([type]): [description]
            is_train (bool, optional): [description]. Defaults to True.
            train_max (int, optional): [description]. Defaults to 81.
        """
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
        for filename in listdir(images_dir):
            image_id = filename[:-4]

            if is_train and int(image_id) >= train_max:
                continue

            if not is_train and int(image_id) < train_max:
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            self.add_image(category, image_id=image_id,
                           path=img_path, annotation=ann_path)
            

    def extract_boxes(self, filename):
        """[summary]

        Args:
            filename ([type]): [description]

        Returns:
            [type]: [description]
        """
        tree = ElementTree.parse(filename)
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
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height


    def create_mask(self, bb, x):
        """[summary]

        Args:
            bb ([type]): [description]
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        rows,cols,*_ = x.shape
        masks = np.zeros((rows, cols))
        bb = bb.astype(np.int)
        masks[bb[0][0]:bb[0][2], bb[0][1]:bb[0][3]] = 1.
        return masks

    def load_mask(self, image_id, image):
        """[summary]

        Args:
            image_id ([type]): [description]

        Returns:
            [type]: [description]
        """
        info = self.image_info[image_id]
        path = info['annotation']

        boxes, w, h = self.extract_boxes(path)
        boxes = np.array(boxes)

        Y = self.create_mask(boxes, image) 

        class_ids = list()
        for i in range(len(boxes)):
            class_ids.append(self.class_names.index('AOP_BTV1'))
            class_ids.append(self.class_names.index('AOP_DIO_01'))
            class_ids.append(self.class_names.index('AOP_EVK80'))
            class_ids.append(self.class_names.index('AOP_TRAS1000'))
            class_ids.append(self.class_names.index('AOP_TRAS1000_no_key'))
            class_ids.append(self.class_names.index('AOP_X10DER_KT_01'))
            class_ids.append(self.class_names.index('SPLITTER_MCP_03'))
            class_ids.append(self.class_names.index('SPLITTER_POA_01IEC'))
            class_ids.append(self.class_names.index('SPLITTER_POA_01_met_kapje'))
            class_ids.append(self.class_names.index('SPLITTER_POA_01_zonder_kapje'))
            class_ids.append(self.class_names.index('SPLITTER_POA_3_met_kapje'))
            class_ids.append(self.class_names.index('SPLITTER_POA_3_zonder_kapje'))
            class_ids.append(self.class_names.index('SPLITTER_SQ601_met_kapje'))
            class_ids.append(self.class_names.index('SPLITTER_UMU_met_kapje'))
            class_ids.append(self.class_names.index('WCD_tweegats'))
        
        return boxes, Y


    def image_reference(self, image_id):
        """[summary]

        Args:
            image_id ([type]): [description]

        Returns:
            [type]: [description]
        """
        info = self.image_info[image_id]
        return info['path']