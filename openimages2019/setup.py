import openimages2019.utils as ut
import pandas as pd
import os
import pyodbc
import numpy as np
from skimage.draw import rectangle
import sys

# Import Mask RCNN
sys.path.append(ut.ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#Make GPUs visible
os.system('export HIP_VISIBLE_DEVICES=0,1,2,3')

def _get_sql_conn():

    if 'dgs_sandbox_pwd' in os.environ:
        conn = pyodbc.connect('DSN=BIdevDatabase;'
                          'Database=Sandbox;'
                          'UID=DGSuser;'
                          'PWD=' + os.environ['dgs_sandbox_pwd'])
    else:
        conn = pyodbc.connect('Driver={SQL Server};Server=bidev;Database=sandbox;Trusted_Connection=yes')

    return conn


def load_classes(conn=None):

    if conn is None:
        conn = _get_sql_conn()

    class_descriptions = pd.read_sql("SELECT LabelName, LabelDescription from [kaggle].[Class_Description]", conn)

    #add 1 since Background class is automatically added at index 0
    class_descriptions['LabelID'] = class_descriptions.index + 1

    return class_descriptions


def load_annotations_by_image():

    conn = _get_sql_conn()
    class_descriptions = load_classes(conn)

    #This now holds the annotation information.
    bboxes = pd.read_sql("SELECT ImageID, XMax, XMin, YMin, YMax, LabelName FROM [Sandbox].[kaggle].[Combined_Set_Detection_BBox]", conn)

    annotations = pd.merge(bboxes,class_descriptions, on='LabelName',how='inner')

    # This now holds the list of images
    image_paths = pd.read_sql("SELECT ImageID, RelativePath, Height, Width, Mode from [kaggle].[Image_Path]", conn)

    # Inner join on the two dataframes, so we now have images combined with associated annotations
    annotated_image_paths = pd.merge(image_paths,annotations, on='ImageID',how='inner')

    return annotated_image_paths


#Subclass the mrcnn DataSet class
class FullKaggleImageDataset(utils.Dataset):

    def add_classes(self, class_descriptions):
        # Add classes, BG is automatically added at index 0
        for _,row in class_descriptions.iterrows():
            self.add_class("openImages", row['LabelID'], row['LabelDescription'])

    def load_kaggle_images(self, dataset_dir, grouped_by_images):
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

    def image_reference(self, image_id):
        return self.image_info[image_id]['path']
    


def load_dataset(anns_by_image, data_dir, classes, is_train=True):

    set_str = 'train' if is_train else 'validation'

    train_paths = anns_by_image[anns_by_image['RelativePath'].str.contains(set_str,regex=False)]
    train_grouped = train_paths.groupby('ImageID')

    # Training dataset
    ds = FullKaggleImageDataset()
    ds.add_classes(classes)
    ds.load_kaggle_images(data_dir, train_grouped)
    ds.prepare()

    return ds

