import os
from mrcnn.config import Config
from mrcnn import utils

ROOT_DIR = os.path.abspath("../../")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

DATA_DIR = os.path.join(ROOT_DIR, "../data")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

#Subclass the mrcnn Config class
class KaggleConfig(Config):
    
    NAME = "kaggle"

    GPU_COUNT = 2
    IMAGES_PER_GPU = 2 # we should be able to make this bigger, atleast 4 for 1024x1024

    # Number of classes (including background)
    # NUM_CLASSES = 1 + len(load_classes())  # + 1 for background
    NUM_CLASSES = 501 # + 1 for background

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


def eval_mAP(model, dataset, config=InferenceConfig(), sample_size=50):

    image_ids = np.random.choice(dataset.image_ids, 500)

    #each input is a tuple of form : image, image_meta, gt_class_id, gt_bbox, gt_mask
    inputs = [modellib.load_image_gt(dataset, config, iid, use_mini_mask=False) for iid in image_ids]

    APs = []

    n = config.BATCH_SIZE

    for i in range(0,len(image_ids),n): 

        curr_inputs = inputs[i:i+n]
        
        results = model.detect([inp[0] for inp in curr_inputs], verbose=0)
        
        for j in range(len(results)):
            r = results[j]
            # Compute AP
            AP, precisions, recalls, overlaps = utils.compute_ap(curr_inputs[j][3], curr_inputs[j][2], curr_inputs[j][4], 
                                                r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
        
    return np.mean(APs)