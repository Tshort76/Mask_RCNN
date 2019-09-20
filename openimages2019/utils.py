from mrcnn import utils
import numpy as np
import mrcnn.model as modellib


def eval_mAP(model, dataset, config, sample_size=50):

    image_ids = np.random.choice(dataset.image_ids, sample_size)

    #each input is a tuple of form : image, image_meta, gt_class_id, gt_bbox, gt_mask
    inputs = [modellib.load_image_gt(dataset, config, iid, use_mini_mask=False) for iid in image_ids]

    APs = []

    n = config.BATCH_SIZE

    for i in range(0,len(image_ids),n): 

        curr_inputs = inputs[i:i+n]
        
        if (len(curr_inputs)%n) != 0:
            break
        
        results = model.detect([inp[0] for inp in curr_inputs], verbose=0)
        
        for j in range(len(results)):
            r = results[j]
            # Compute AP
            AP, precisions, recalls, overlaps = utils.compute_ap(curr_inputs[j][3], curr_inputs[j][2], curr_inputs[j][4], 
                                                r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
        
    return np.mean(APs)