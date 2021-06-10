## Inference.ipynb
In inference.ipynb notebook, you'll need to follow 4 steps in order to inference your model
1. Prepare prerequisite
2. Model Preparation 
3. Detection
4. Inference
 
### **1. Prepare prerequisite** 
 
#### **1.1 Installation**

You can install the TensorFlow Object Detection API either with Python Package Installer (pip) or Docker. For local runs I recommend using Docker, for Google Colab and Anaconda Envs, I recommend using pip. You can check full docs at [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

Clone tensorflow official repo. This contains API installation files, training/evaluation script, exporting inference model script and more. 

```bash
!git clone --depth 1 https://github.com/tensorflow/models
```

Python Package Installation 

```bash
!sudo apt install -y protobuf-compiler
%cd models/research
!protoc object_detection/protos/*.proto --python_out=.
!cp object_detection/packages/tf2/setup.py .
!python -m pip install .
```

To test the installation run:

```python
# Test the installation.
!python models/research/object_detection/builders/model_builder_tf2_test.py
```

If everything installed correctly you should see something like:

```bash
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 91.767s

OK (skipped=1)
```
#### **1.2 Import Dependencies**

Note: You can only import object detection related modules only after you've installed tensorflow object detection API

```python
import numpy as np
import cv2
import os
import sys
import time
import tensorflow as tf
import math
import pathlib
import math
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from IPython.display import HTML
from base64 import b64encode
import time

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
```

#### **1.3 Download my pre-trained model or import your model**

You can either use your own inference graph or [mine](https://github.com/NyanSwanAung/Pothole-Detection-using-MaskRCNN/releases/download/v1.0/) to detect on images and video.

```bash
# Download my inferenced graph from github release page 
!wget https://github.com/NyanSwanAung/Pothole-Detection-using-MaskRCNN/releases/download/v1.0/inference_graph.zip
!unzip inference_graph.zip

# Remove unecessary files
!rm -r __MACOSX
!rm -r inference_graph.zip
```

#### **1.4 Download test dataset and label map**

In order to detect your model on images and video, you'll need test dataset and label map.
```bash
# Test dataset
!wget https://github.com/NyanSwanAung/Pothole-Detection-using-MaskRCNN/releases/download/v1.0/test.zip
!unzip test.zip 

# Label Map
!wget https://github.com/NyanSwanAung/Pothole-Detection-using-MaskRCNN/releases/download/v1.0/label.pbtxt

# Remove unecessary files
!rm -r __MACOSX
!rm -r test.zip
```

After successful installation and downloads, you'll get a directory structure like below.

```
inference_graph
    └── checkpoint
    └── saved_model
    └── pipeline.config
models
test
  └── sample1.jpg
  └── sample2.jpg
  └── sample3.jpg
  └── sample4.jpg
  └── sample5.jpg
  └── sample6.jpg
  └── sample7.jpg
  └── test_vid.mp4
label.pbtxt
```

### **2. Model Preparation** 

#### **2.1 Setup Paths**
```python
IG_PATH = '/content/inference_graph'
LABEL_MAP_PATH = '/content/label.pbtxt'
TEST_IMG_DIR = '/content/test'
TEST_VID_DIR = '/content/test/test_vid.mp4'
```

#### **2.2 Load Model**

A function to load our inference graph. It will also show how long it takes to load inference graph.
```python
def load_model():
    start = time.time()
    model_dir = IG_PATH
    model_dir = pathlib.Path(model_dir)/"saved_model"
    model = tf.saved_model.load(str(model_dir))
    end = time.time()
    total = math.ceil(end-start)
    print(f'It took {total}s to load model')
    return model
```


#### **2.3 Load Label Map and Test Dataset**

In order to detect on images and video, you'll need to get test dataset images path and load your dataset label map which contains id and name. 

```python
# List of the strings that is used to add correct label for each box.
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path(TEST_IMG_DIR)
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
```


#### **2.4 Detection**
```python
# Load inference graph and model
detection_model = load_model()
```
In order to feed our test images, you'll need to do post-processing steps.

```python 
def run_inference_for_single_image(model, image):
    
    image = np.asarray(image)
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    
    num_detections = int(output_dict.pop('num_detections'))
    need_detection_key = ['detection_classes','detection_boxes','detection_masks','detection_scores']
    output_dict = {key: output_dict[key][0, :num_detections].numpy()
                   for key in need_detection_key}
    
    output_dict['num_detections'] = num_detections
    
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            tf.convert_to_tensor(output_dict['detection_masks']), output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict
```

A function to inference on test images by converting image to tensor type, using trained model to draw bbox on potholes 
```python 
def run_inference_image(model, image_path):
  
  start = time.time()
  
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', True),
      use_normalized_coordinates=True,
      line_thickness=5)
   
  end = time.time()
  total = math.ceil(end-start)
  display(Image.fromarray(image_np))
  print(f'It took {total}s for above image')
```

A function to inference on test video by loading the video, detect on each frame using our trained model, rewriting the drawn bounding boxes as new frame and save new frames as a new video called 'detected_output.avi'. You can see the detected video in 
[results folder](https://github.com/NyanSwanAung/Pothole-Detection-using-MaskRCNN/results)
```python
def run_inference_video(model, video_path):

  cap = cv2.VideoCapture(video_path)

  if cap.isOpened():
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      res=(int(width), int(height))

      # save detected video
      # Initialize our video writer
      fourcc = cv2.VideoWriter_fourcc(*"XVID") #codec
      out = cv2.VideoWriter('detected_output.avi', fourcc, 20.0, res)
      frame = None

      while True:
          try:
              is_success, image_np = cap.read()
          except cv2.error:
              continue

          if not is_success:
              break

          # Actual detection.
          start = time.time()
          image_np = np.array(image_np)
            
          # Actual detection.
          output_dict = run_inference_for_single_image(model, image_np)

          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks_reframed', None),
              use_normalized_coordinates=True,
              line_thickness=8)
          
          end = time.time()
          total = math.ceil(end-start)
          print(f'{total}s per frame')
          out.write(image_np)
        
      out.release() 

      # OPTIONAL: show last image
      if frame:
        cv2_imshow(frame)

  cap.release()

```
### **4. Inference**

#### **4.1 Inference on Image**
```python 
for image_path in TEST_IMAGE_PATHS:
  run_inference_image(detection_model, image_path)
```

#### **4.2 Inference on Video**

After running this code, you'll get detected_output.avi which contains bounding boxes and masks of potholes.
```python
# Inference on Video
run_inference_video(detection_model, TEST_VID_DIR)
```

#### **4.2 Play video in colab**

For your convenience, you can play your detected_output.avi in colab cell using below code. This will create a new compressed_output.mp4.

```python
# detected video path
input_path = 'detected_output.avi'

# Compressed video path
compressed_path = "/content/compressed_output.mp4"

os.system(f"ffmpeg -i {input_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
```

