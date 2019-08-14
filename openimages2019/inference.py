import numpy as np
from mrcnn import utils
import mrcnn.model as modellib
import openimages2019.utils as ut

class InferenceConfig(ut.KaggleConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    BATCH_SIZE = 10



# Recreate the model in inference mode
# model = modellib.MaskRCNN(mode="inference", 
#                           config=inference_config,
#                           model_dir=ut.MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = model.find_last()

# Load trained weights

# inference_model(model_path)
def inference_model(model_path, config=InferenceConfig()):
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ut.MODEL_DIR)
    
    model.load_weights(model_path, by_name=True)

    return model


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
