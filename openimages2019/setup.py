import openimages2019.utils as ut
import pandas as pd
import os
import pyodbc
import numpy as np
from skimage.draw import rectangle
from skimage import transform
import skimage.io
import sys

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


def _get_sql_conn():
    return pyodbc.connect(os.environ['db_connection_string'])


def load_classes(path_to_csv=None) -> pd.DataFrame:
    """Loads class labels and their descriptions from the kaggle database (for object detection track only) or from a csv file
    
    Keyword Arguments:
        path_to_csv {str} -- file path for the csv containing class labels and description (default: {None})
    
    Returns:
        pandas.DataFrame -- Dataframe containing the class name, class description, and labelId (updated to account for implicit background class)
    """

    conn = _get_sql_conn()

    if path_to_csv:
        class_descriptions = pd.read_csv(path_to_csv)
    else: #assume that we are using object detection classes
        class_descriptions = pd.read_sql("SELECT LabelName, LabelDescription from [kaggle].[Class_Description]", conn)

    #add 1 since Background class is automatically added at index 0
    class_descriptions['LabelID'] = class_descriptions.index + 1

    return class_descriptions


def load_annotations_by_image(classes=None, use_masks=False) -> pd.DataFrame:
    """Loads and returns ground truth annotations for a set of classes.

    
    Keyword Arguments:
        classes {pd.DataFrame} -- The set of classes of interest.  Annotations are filtered to those involving one of the classes (default: {None})
        use_masks {bool} -- Whether or not to use segmentation masks as the ground truth (default: {False})
    
    Returns:
        pd.DataFrame -- A dataframe in which each row represents a single annotated object within an image
    """

    conn = _get_sql_conn()
    
    if classes is None:
        classes  = load_classes(conn)

    if use_masks:
        bboxes = pd.read_sql("SELECT MaskPath, ImageID, LabelName, SourceDataset from [kaggle].[Combined_Annotations_Object_Segmentation]", conn)
    else:
        bboxes = pd.read_sql("SELECT ImageID, XMax, XMin, YMin, YMax, LabelName FROM [kaggle].[Combined_Set_Detection_BBox]", conn)

    annotations = pd.merge(bboxes,classes, on='LabelName',how='inner')

    # This now holds the list of images
    image_paths = pd.read_sql("SELECT ImageID, RelativePath, Height, Width, Mode from [kaggle].[Image_Path]", conn)

    # Inner join on the two dataframes, so we now have images combined with associated annotations
    annotated_image_paths = pd.merge(image_paths,annotations, on='ImageID',how='inner')

    return annotated_image_paths


#Subclass the mrcnn DataSet class
class OpenImageDataset(utils.Dataset):

    def set_mask_path(self, path=None):
        self.mask_path = path

    def add_classes(self, class_descriptions):
        # Add classes, BG is automatically added at index 0 so LabelID has been modified to start at 1
        for _,row in class_descriptions.iterrows():
            self.add_class("openImages", row['LabelID'], row['LabelDescription'])

    def load_image_files(self, dataset_dir, grouped_by_images):
        """Load a subset of the image dataset.
        dataset_dir: The root directory of the image dataset.
        classes: Dataframe : If provided, only loads images that have the given classes.
        """
        
        # Add images
        for i,df in grouped_by_images:    
            row = df.iloc[0]
            
            self.add_image(
                "openImages", image_id=i,
                path=os.path.join(dataset_dir, row['RelativePath']),
                width=row["Width"],
                height=row["Height"],
                annotations=df)
    

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        
        if self.mask_path:
            return self._load_file_masks(image_id)

        return self._build_bbox_mask(image_id)
        

    def _build_bbox_mask(self, image_id):
        # Create rectangular bounding box since we are doing object detection, not segmentation
        # desired dimension is [height, width, instance_count]
        img = self.image_info[image_id]
        
        mask = np.zeros([img["height"], img["width"], len(img["annotations"])],
                        dtype=np.uint8)
        
        for i,(_,p) in enumerate(img["annotations"].iterrows()):
            # Create rectangular bounding box since we are doing object detection, not segmentation
    
            xmax = int(img["width"]*p['XMax'])
            xmin = int(img["width"]*p['XMin'])
            ymin = int(img["height"]*p['YMin'])
            ymax = int(img["height"]*p['YMax'])
            
            start = (ymin, xmin)  #top left corner ... are coordinates reversed?
            end = (ymax, xmax)  #height and width
            rr, cc = rectangle(start, end=end, shape=(img["height"],img["width"]))
            
            mask[rr, cc, i] = 1

        
        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.array(img['annotations']['LabelID'].values, dtype=np.int32)

    def _load_file_masks(self, image_id):
        img = self.image_info[image_id]
        
        masks = []
        
        for _,p in img["annotations"].iterrows():   
            mpath = os.path.join(self.mask_path, p['MaskPath'])
            raw_mask = skimage.io.imread(mpath)
             # mask is often not the same size as the image, resize the mask so that it is
            masks.append(transform.resize(raw_mask,(img['height'],img['width'])))
        
        mask = np.stack(masks, axis=-1)
        
        return mask,np.array(img['annotations']['LabelID'].values, dtype=np.int32)


    def image_reference(self, image_id):
        return self.image_info[image_id]['path']
    


def load_dataset(anns_by_image, data_dir, classes, is_train=True, mask_path=None):
    """Prepares and returns an OpenImageDataset for the OpenImages2019 image set
    
    Arguments:
        anns_by_image {pd.DataFrame} -- A dataframe in which each row represents a single annotated object within an image
        data_dir {str} -- The filepath to where the images are stored.  Object detection assumes a 'validation' and 'train' subdirectory at this location
        classes {pd.DataFrame} -- The classes of interest for this dataset
    
    Keyword Arguments:
        is_train {bool} -- Load the training set (as opposed to the validation set)? (default: {True})
        mask_path {str} -- The filepath to where the segmentation masks are stored (for object segmentation) (default: {None})
    
    Returns:
        mrcnn.utils.Dataset -- The mrcnn dataset expected for training the matterport mask_rcnn model
    """

    set_str = 'train' if is_train else 'validation'

    if mask_path:
        ann_paths = anns_by_image[anns_by_image['SourceDataset'] == set_str]
    else:
        ann_paths = anns_by_image[anns_by_image['RelativePath'].str.contains(set_str,regex=False)]

    anns_grouped = ann_paths.groupby('ImageID')

    ds = OpenImageDataset()
    ds.add_classes(classes)
    ds.set_mask_path(mask_path)
    ds.load_image_files(data_dir, anns_grouped)
    ds.prepare()

    return ds
