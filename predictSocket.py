import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os

CLASS_NAMES = ['BG','AOP_BTV1','AOP_DIO_01','AOP_EVK80',
                'AOP_TRAS1000','AOP_TRAS1000_no_key',
                'AOP_X10DER_KT_01', 'SPLITTER_MCP_03',
                'SPLITTER_POA_01IEC','SPLITTER_POA_01_met_kapje',
                'SPLITTER_POA_01_zonder_kapje', 'SPLITTER_POA_3_met_kapje',
                'SPLITTER_POA_3_zonder_kapje', 'SPLITTER_SQ601_met_kapje',
                'SPLITTER_UMU_met_kapje', 'WCD_tweegats']

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)

model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

model.load_weights(filepath="/Users/wolfsinem/maskrcnn/multiClassWeights/mask_rcnn_socketconfig_0001.h5", 
                   by_name=True)

image = cv2.imread("/Users/wolfsinem/Downloads/socketdata/Images/00666.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = model.detect([image], verbose=0)
r = r[0]

mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])