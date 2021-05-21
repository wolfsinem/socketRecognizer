
from numpy import expand_dims
from numpy import mean

from mrcnn.config import Config
from mrcnn.model import MaskRCNN

from mrcnn.utils import compute_ap, compute_recall
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

from trainSocket import SocketDataset

class PredictionConfig(Config):
	NAME = "socket_cfg"
	NUM_CLASSES = 1 + 15
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

def evaluate_model(dataset, model, cfg):
  """[summary]

  Args:
      dataset ([type]): [description]
      model ([type]): [description]
      cfg ([type]): [description]

  Returns:
      [type]: [description]
  """
  APs = list()
  # ARs = list()
  # F1_scores = list()
  for image_id in dataset.image_ids:
      image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
      scaled_image = mold_image(image, cfg)
      sample = expand_dims(scaled_image, 0)
      yhat = model.detect(sample, verbose=0)
      r = yhat[0]
      
      AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
      # AR, positive_ids = compute_recall(r["rois"], gt_bbox, iou=0.2)
      
      # ARs.append(AR)
      # F1_scores.append((2* (mean(precisions) * mean(recalls)))/(mean(precisions) + mean(recalls)))
      APs.append(AP)
  
  mAP = mean(APs)
  # mAR = mean(ARs)
  return mAP


dataset_dir = '/Users/wolfsinem/Downloads/socketdata/'

test_set = SocketDataset()
test_set.load_dataset(dataset_dir=dataset_dir, is_train=False)
test_set.prepare()

eval_config = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir=dataset_dir, config=eval_config)
model.load_weights('/Users/wolfsinem/maskrcnn/multiClassWeights/mask_rcnn_socketconfig_0001.h5', by_name=True)

mAP = evaluate_model(test_set, model, eval_config)
print("mAP (mean average precision): %.4f" % mAP)
# print("mAR: (mean average recall): %.4f" % mAR)

# F1_score_2 = (2 * mAP * mAR)/(mAP + mAR)
# print('F1-score : ', F1_score_2)