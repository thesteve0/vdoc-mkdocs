# Drawing Labels on Samples [¶](\#Drawing-Labels-on-Samples "Permalink to this headline")

This recipe demonstrates how to use FiftyOne to render annotated versions of image and video [samples](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#samples) with their [label field(s)](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#labels) overlaid.

## Setup [¶](\#Setup "Permalink to this headline")

If you haven’t already, install FiftyOne:

```python
[ ]:

```

```python
!pip install fiftyone

```

In this recipe we’ll use the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/zoo_datasets.html) to download some labeled datasets to use as sample data for drawing labels.

Behind the scenes, FiftyOne uses either the [TensorFlow Datasets](https://www.tensorflow.org/datasets) or [TorchVision Datasets](https://pytorch.org/vision/stable/datasets.html) libraries to wrangle the datasets, depending on which ML library you have installed.

You can, for example, install PyTorch as follows (we’ll also need `pycocotools` to load the COCO dataset, in particular):

```python
[1]:

```

```python
!pip install torch torchvision
!pip install pycocotools

```

## Drawing COCO detections [¶](\#Drawing-COCO-detections "Permalink to this headline")

You can download the validation split of the COCO-2017 dataset to `~/fiftyone/coco-2017/validation` by running the following command:

```python
[1]:

```

```python
!fiftyone zoo datasets download coco-2017 --splits validation

```

```python
Split 'validation' already downloaded

```

Now let’s load the dataset, extract a [DatasetView](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#datasetviews) that contains 100 images from the dataset, and render them as annotated images with their ground truth labels overlaid:

```python
[2]:

```

```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.annotations as foua

# Directory to write the output annotations
anno_dir = "/tmp/fiftyone/draw_labels/coco-2017-validation-anno"

# Load the validation split of the COCO-2017 dataset
dataset = foz.load_zoo_dataset("coco-2017", split="validation")

# Extract some samples
view = dataset.limit(100)

#
# You can customize the look-and-feel of the annotations
# For more information, see:
# https://voxel51.com/docs/fiftyone/user_guide/draw_labels.html#customizing-label-rendering
#
config = foua.DrawConfig({
    "per_object_label_colors": True
})

# Render the labels
print("Writing annotated images to '%s'" % anno_dir)
view.draw_labels(anno_dir, config=config)
print("Annotation complete")

```

```python
Split 'validation' already downloaded
Loading 'coco-2017' split 'validation'
 100% |█████| 5000/5000 [14.8s elapsed, 0s remaining, 339.4 samples/s]
Writing annotated images to '/tmp/fiftyone/draw_labels/coco-2017-validation-anno'
 100% |███████| 100/100 [7.3s elapsed, 0s remaining, 11.9 samples/s]
Annotation complete

```

Let’s list the output directory to verify that the annotations have been generated:

```python
[3]:

```

```python
!ls -lah /tmp/fiftyone/draw_labels/coco-2017-validation-anno | head

```

```python
total 51976
drwxr-xr-x  202 Brian  wheel   6.3K Jul 27 18:36 .
drwxr-xr-x    5 Brian  wheel   160B Jul 27 15:59 ..
-rw-r--r--    1 Brian  wheel   115K Jul 27 18:36 000001-2.jpg
-rw-r--r--@   1 Brian  wheel   116K Jul 27 12:51 000001.jpg
-rw-r--r--    1 Brian  wheel   243K Jul 27 18:36 000002-2.jpg
-rw-r--r--    1 Brian  wheel   243K Jul 27 12:51 000002.jpg
-rw-r--r--    1 Brian  wheel   177K Jul 27 18:36 000003-2.jpg
-rw-r--r--@   1 Brian  wheel   177K Jul 27 12:51 000003.jpg
-rw-r--r--    1 Brian  wheel   101K Jul 27 18:36 000004-2.jpg

```

Here’s an example of an annotated image that was generated:

![coco-2017-annotated](../_images/draw_labels_coco2017.jpg)

## Drawing Caltech 101 classifications [¶](\#Drawing-Caltech-101-classifications "Permalink to this headline")

You can download the test split of the Caltech 101 dataset to `~/fiftyone/caltech101/test` by running the following command:

```python
[4]:

```

```python
!fiftyone zoo datasets download caltech101 --splits test

```

```python
Split 'test' already downloaded

```

Now let’s load the dataset, extract a [DatasetView](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#datasetviews) that contains 100 images from the dataset, and render them as annotated images with their ground truth labels overlaid:

```python
[5]:

```

```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.annotations as foua

# Directory to write the output annotations
anno_dir = "/tmp/fiftyone/draw_labels/caltech101-test-anno"

# Load the test split of the Caltech 101 dataset
dataset = foz.load_zoo_dataset("caltech101", split="test")

# Extract some samples
view = dataset.limit(100)

#
# You can customize the look-and-feel of the annotations
# For more information, see:
# https://voxel51.com/docs/fiftyone/user_guide/draw_labels.html#customizing-label-rendering
#
config = foua.DrawConfig({
    "font_size": 36
})

# Render the labels
print("Writing annotated images to '%s'" % anno_dir)
view.draw_labels(anno_dir, config=config)
print("Annotation complete")

```

```python
Split 'test' already downloaded
Loading 'caltech101' split 'test'
 100% |█████| 9145/9145 [4.8s elapsed, 0s remaining, 1.9K samples/s]
Writing annotated images to '/tmp/fiftyone/draw_labels/caltech101-test-anno'
 100% |███████| 100/100 [2.6s elapsed, 0s remaining, 37.4 samples/s]
Annotation complete

```

Let’s list the output directory to verify that the annotations have been generated:

```python
[6]:

```

```python
!ls -lah /tmp/fiftyone/draw_labels/caltech101-test-anno | head

```

```python
total 17456
drwxr-xr-x  182 Brian  wheel   5.7K Jul 27 18:37 .
drwxr-xr-x    5 Brian  wheel   160B Jul 27 15:59 ..
-rw-r--r--@   1 Brian  wheel    13K Jul 27 18:37 image_0001-2.jpg
-rw-r--r--    1 Brian  wheel    41K Jul 27 15:59 image_0001.jpg
-rw-r--r--    1 Brian  wheel   197K Jul 27 18:37 image_0002.jpg
-rw-r--r--    1 Brian  wheel   5.9K Jul 27 18:37 image_0003.jpg
-rw-r--r--    1 Brian  wheel    19K Jul 27 18:37 image_0004-2.jpg
-rw-r--r--    1 Brian  wheel    33K Jul 27 15:59 image_0004.jpg
-rw-r--r--    1 Brian  wheel    18K Jul 27 18:37 image_0005-2.jpg

```

Here’s an example of an annotated image that was generated:

![49a6a3c719a8439cbf418c6117a2cb2a](../_images/draw_labels_caltech101.jpg)

## Drawing labels on videos [¶](\#Drawing-labels-on-videos "Permalink to this headline")

FiftyOne can also render frame labels onto video samples.

To demonstrate, let’s work with the (small) video quickstart dataset from the zoo:

```python
[2]:

```

```python
import fiftyone.zoo as foz

# Load a small video dataset
dataset = foz.load_zoo_dataset("quickstart-video")

print(dataset)

```

```python
Dataset already downloaded
Loading 'quickstart-video'
 100% |█████████| 10/10 [15.9s elapsed, 0s remaining, 0.6 samples/s]
Name:           quickstart-video
Media type      video
Num samples:    10
Persistent:     False
Info:           {'description': 'quickstart-video'}
Tags:           []
Sample fields:
    media_type: fiftyone.core.fields.StringField
    filepath:   fiftyone.core.fields.StringField
    tags:       fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:   fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    frames:     fiftyone.core.fields.FramesField
Frame fields:
    frame_number: fiftyone.core.fields.FrameNumberField
    objs:         fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)

```

Note that the dataset contains frame-level detections in the `objs` field of each frame.

Let’s make a [DatasetView](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#datasetviews) that contains a couple random videos from the dataset and render them as annotated videos with the frame-level detections overlaid:

```python
[3]:

```

```python
import fiftyone.utils.annotations as foua

# Directory to write the output annotations
anno_dir = "/tmp/fiftyone/draw_labels/quickstart-video-anno"

# Extract two random samples
view = dataset.take(2)

#
# You can customize the look-and-feel of the annotations
# For more information, see:
# https://voxel51.com/docs/fiftyone/user_guide/draw_labels.html#customizing-label-rendering
#
config = foua.DrawConfig({
    "per_object_label_colors": True
})

# Render the labels
print("Writing annotated videos to '%s'" % anno_dir)
view.draw_labels(anno_dir, config=config)
print("Annotation complete")

```

```python
Writing annotated videos to '/tmp/fiftyone/draw_labels/quickstart-video-anno'
Rendering video 1/2: '/tmp/fiftyone/draw_labels/quickstart-video-anno/0587e1cfc2344523922652d8b227fba4-000014-video_052.mp4'
 100% |████████| 120/120 [19.0s elapsed, 0s remaining, 6.7 frames/s]
Rendering video 2/2: '/tmp/fiftyone/draw_labels/quickstart-video-anno/0587e1cfc2344523922652d8b227fba4-000014-video_164.mp4'
 100% |████████| 120/120 [27.2s elapsed, 0s remaining, 4.3 frames/s]
Annotation complete

```

Let’s list the output directory to verify that the annotations have been generated:

```python
[4]:

```

```python
!ls -lah /tmp/fiftyone/draw_labels/quickstart-video-anno

```

```python
total 34832
drwxr-xr-x  4 Brian  wheel   128B Oct  7 23:57 .
drwxr-xr-x  3 Brian  wheel    96B Oct  7 23:57 ..
-rw-r--r--  1 Brian  wheel   7.5M Oct  7 23:57 0587e1cfc2344523922652d8b227fba4-000014-video_052.mp4
-rw-r--r--  1 Brian  wheel   8.5M Oct  7 23:58 0587e1cfc2344523922652d8b227fba4-000014-video_164.mp4

```

Here’s a snippet of an annotated video that was generated:

![quickstart-video-annotated](../_images/draw_labels_quickstart_video.gif)

## Cleanup [¶](\#Cleanup "Permalink to this headline")

You can cleanup the files generated by this recipe by running the command below:

```python
[7]:

```

```python
!rm -rf /tmp/fiftyone

```

