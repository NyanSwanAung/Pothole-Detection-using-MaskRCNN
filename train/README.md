## Train.ipynb 
In train.ipynb notebook, you'll need to follow 5 steps in order to train the model
1. Prepare prerequisite
2. Setup Paths
3. Config for fine-tuning pothole dataset
4. Training and Evaluation 
5. Export model to inference graph 

### 1. Prepare prerequisites

#### **1.1 Download pre-trained model and config file**

Download Mask-RCNN model from [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), extract the zip file and move in **pre-trained-models** folder

```
from IPython.display import clear_output

# Download using wget
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz

# Extract downloaded tar file 
!tar -xf mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz

os.mkdir('pre-trained-models')

# Move extracted folder to pretrained-model
!mv mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8 pre-trained-models
```

#### **1.2 Installation**

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
#### **1.3 Import Dependencies**

Note: You can only import object detection related modules only after you've installed tensorflow object detection API

```python
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
```

#### **1.4 Pothole Dataset**

The original dataset is from this [link](https://www.kaggle.com/atulyakumar98/pothole-detection-dataset) which are unannotated.
If you want to label your custom data, there are various online tools such as for object detection, you can use [LabelImg](https://github.com/tzutalin/labelImg), an excellent image annotation tool supporting both PascalVOC and Yolo format. For Image Segmentation / Instance Segmentation there are multiple great annotations tools available. Including, [VGG Image Annotation Tool](http://www.robots.ox.ac.uk/~vgg/software/via/), [labelme](https://github.com/wkentaro/labelme), and [PixelAnnotationTool](https://github.com/abreheret/PixelAnnotationTool). 

Once you've annotated, you'll probably get different dataset formats (VOC XML or COCO JSON). Make sure you convert these formats to .tfrecord format if you're using TF-OD-API. There are many gists for this dataset format conversion.

In order to train with TF-OD-API, we need the dataset as .tfrecord or .record file format. Simply you'll need 
1. train.tfrecord
2. valid.tfrecord
3. label_map.pbtxt 

If you're new to TF Record, I recommend this great [article](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564) from medium. 

However, I chose to use annotated dataset from this [repo](https://github.com/SamdenLepcha/Pothole-Detection-With-Mask-R-CNN/tree/master/place_in_object_detection) by SamdenLepcha.

```python
# download train.record
!wget https://raw.githubusercontent.com/SamdenLepcha/Pothole-Detection-With-Mask-R-CNN/master/place_in_object_detection/train.record

# download test.record
!wget https://raw.githubusercontent.com/SamdenLepcha/Pothole-Detection-With-Mask-R-CNN/master/place_in_object_detection/test.record

# download label map
!wget https://raw.githubusercontent.com/SamdenLepcha/Pothole-Detection-With-Mask-R-CNN/master/place_in_object_detection/training/label.pbtxt
```

#### **1.5 Make Workspace**
As recommended by [official documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#preparing-the-workspace), we're going to create this directory structure. Note: you can still train your model if you don't have strucutre like this. You just need to configure right paths.

```
models
workspace
  └── annotations
         └── label.pbtxt
         └── train.record
         └── valid.record
  └── models
         └── my_maskrcnn
                └── pipeline.config
  └── pre-trained-models
         └── mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8
                └── checkpoint
                └── saved_model
                └── pipeline.config
```

Run below code to create this kind of structure

```bash
%cd ~/../content
os.mkdir('workspace')
os.mkdir('workspace/models')
os.mkdir('workspace/models/my_maskrcnn')

%cd ~/../content
os.mkdir('annotations')
!mv train.record label.pbtxt -t annotations 

# Rename test.record to valid.record
!mv test.record valid.record
!mv valid.record -t annotations
!mv annotations pre-trained-models -t workspace
!cp /content/workspace/pre-trained-models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/pipeline.config workspace/models/my_maskrcnn 
```

### 2. Setup Paths

```python
WORKSPACE_PATH = 'workspace'
APIMODEL_PATH = 'models'

ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'

MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'

CUSTOM_MODEL_NAME = 'my_maskrcnn'

CONFIG_PATH = MODEL_PATH+'/' + CUSTOM_MODEL_NAME + '/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/' + CUSTOM_MODEL_NAME + '/'
```

### 3. Update config file for Pothole Dataset

Sometimes the default config file which comes with pre-trained-model, contains typos. So that, we're going to download maskrcnn [config file](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config) from official repo which is the updatest. 

```bash
%cd /content/workspace/models/my_maskrcnn
!wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config

# Rename to pipeline.config
!mv mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config pipeline.config
%cd ~/../content
```

Once we have the updated config, you'll need to load and read config file to rewrite custom configs for custom dataset.
```python
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
  proto_str = f.read()
  text_format.Merge(proto_str, pipeline_config)
```

You can manually rewrite config by opening the file in colab or jupyter. Moreoever, you can also use below code to customize configs.

```python
# Rewriting 

# Hyperparameters 
pipeline_config.model.faster_rcnn.num_classes = 1
pipeline_config.train_config.batch_size = 1
pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = 512
pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = 512
pipeline_config.train_config.num_steps = 3000

# pre-trained model's checkpoint path
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/checkpoint/ckpt-0'

# pre-trained model's checkpoint type
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"

# Train label map path and record path 
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']

# Valid label map path and record path
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/valid.record']

# Shuffle valid dataset
pipeline_config.eval_input_reader[0].shuffle = True

# Save config file
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
    f.write(config_text)  
```

### 4. Train and Evaluate Model

**To train the model**, you'll need to run *models/research/object_detection/model_main_tf2.py* from official github repo with required arguments.
If you want to see supported arguments, run below code.

```bash
# To see available arguments
!python models/research/object_detection/model_main_tf2.py --help
```

I've used default arguments and changed important ones.
```bash
# Run Train Script 
!python models/research/object_detection/model_main_tf2.py \
  --model_dir={CHECKPOINT_PATH} \
  --pipeline_config_path=workspace/models/{CUSTOM_MODEL_NAME}/pipeline.config \
  --alsologtostderr \
```
If you start seeing these outputs, your model is good to go 🚀
```
Use fn_output_signature instead
INFO:tensorflow:Step 100 per-step time 1.375s
I0609 08:00:03.346771 140532679501696 model_lib_v2.py:700] Step 100 per-step time 1.375s
INFO:tensorflow:{'Loss/BoxClassifierLoss/classification_loss': 0.018416926,
 'Loss/BoxClassifierLoss/localization_loss': 0.0,
 'Loss/BoxClassifierLoss/mask_loss': 0.0,
 'Loss/RPNLoss/localization_loss': 0.083752826,
 'Loss/RPNLoss/objectness_loss': 0.66165924,
 'Loss/regularization_loss': 0.0,
 'Loss/total_loss': 0.763829,
 'learning_rate': 0.00016000001}
```
**To evaluate your model** with COCO MAP and COCO MAR, you'll need to run the same training code file but with different arguments. 
```python
# Run evaluation script 
!python models/research/object_detection/model_main_tf2.py \
  --model_dir={CHECKPOINT_PATH} \
  --pipeline_config_path=workspace/models/{CUSTOM_MODEL_NAME}/pipeline.config \
  --checkpoint_dir={CHECKPOINT_PATH}
```
If you start seeing these similar outputs, your model is good to go 🚀
```bash
Accumulating evaluation results...
DONE (t=0.06s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.097
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.268
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.053
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.020
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.096
```
**To see your loss metrics in tensorboard**, run below cell
```
# Load the TensorBoard notebook extension
%load_ext tensorboard
%tensorboard --logdir {CHECKPOINT_PATH}
```

### 4. Export Inference Graph

To evaluate on your test dataset, you'll need to export your tensorflow checkpoint files to inference graph. I've already uploaded my checkpoint files and inference graph in [release page](https://github.com/NyanSwanAung/Pothole-Detection-using-MaskRCNN/releases) if you want to use. 

```
output_directory = CHECKPOINT_PATH + 'inference_graph'
!python models/research/object_detection/exporter_main_v2.py \
    --trained_checkpoint_dir {CHECKPOINT_PATH} \
    --output_directory {output_directory} \
    --pipeline_config_path {CONFIG_PATH}
```












