Table of Contents

- [Docs](../index.html) >

- [FiftyOne Recipes](index.html) >
- Convert Dataset Formats

Contents


# Convert Dataset Formats [¶](\#Convert-Dataset-Formats "Permalink to this headline")

This recipe demonstrates how to use FiftyOne to convert datasets on disk between common formats.

## Setup [¶](\#Setup "Permalink to this headline")

If you haven’t already, install FiftyOne:

```
[ ]:

```

```
pip install fiftyone

```

This notebook contains bash commands. To run it as a notebook, you must install the [Jupyter bash kernel](https://github.com/takluyver/bash_kernel) via the command below.

Alternatively, you can just copy + paste the code blocks into your shell.

```
[1]:

```

```
pip install bash_kernel
python -m bash_kernel.install

```

In this recipe we’ll use the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/zoo_datasets.html) to download some open source datasets to work with.

Specifically, we’ll need [TensorFlow](https://www.tensorflow.org/) and [TensorFlow Datasets](https://www.tensorflow.org/datasets) installed to [access the datasets](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/zoo_datasets.html#customizing-your-ml-backend):

```
[2]:

```

```
pip install tensorflow tensorflow-datasets

```

## Download datasets [¶](\#Download-datasets "Permalink to this headline")

Download the test split of the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) from the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/zoo_datasets.html) using the command below:

```
[1]:

```

```
# Download the test split of CIFAR-10
fiftyone zoo datasets download cifar10 --split test

```

```
Downloading split 'test' to '~/fiftyone/cifar10/test'
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ~/fiftyone/cifar10/tmp-download/cifar-10-python.tar.gz
170500096it [00:04, 35887670.65it/s]
Extracting ~/fiftyone/cifar10/tmp-download/cifar-10-python.tar.gz to ~/fiftyone/cifar10/tmp-download
 100% |███| 10000/10000 [5.2s elapsed, 0s remaining, 1.8K samples/s]
Dataset info written to '~/fiftyone/cifar10/info.json'

```

Download the validation split of the [KITTI dataset](http://www.cvlibs.net/datasets/kitti) from the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/zoo_datasets.html) using the command below:

```
[1]:

```

```
# Download the validation split of KITTI
fiftyone zoo datasets download kitti --split validation

```

```
Split 'validation' already downloaded

```

## The fiftyone convert command [¶](\#The-fiftyone-convert-command "Permalink to this headline")

The [FiftyOne CLI](https://voxel51.com/docs/fiftyone/cli/index.html) provides a number of utilities for importing and exporting datasets in a variety of common (or custom) formats.

Specifically, the `fiftyone convert` command provides a convenient way to convert datasets on disk between formats by specifying the [fiftyone.types.Dataset](https://voxel51.com/docs/fiftyone/api/fiftyone.types.html#fiftyone.types.dataset_types.Dataset) type of the input and desired output.

FiftyOne provides a collection of [builtin types](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#supported-formats) that you can use to read/write datasets in common formats out-of-the-box:

| Dataset format | Import Supported? | Export Supported? | Conversion Supported? |
| --- | --- | --- | --- |
| [ImageDirectory](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#imagedirectory) | ✓ | ✓ | ✓ |
| [VideoDirectory](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#videodirectory) | ✓ | ✓ | ✓ |
| [FiftyOneImageClassificationDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#fiftyoneimageclassificationdataset) | ✓ | ✓ | ✓ |
| [ImageClassificationDirectoryTree](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#imageclassificationdirectorytree) | ✓ | ✓ | ✓ |
| [TFImageClassificationDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#tfimageclassificationdataset) | ✓ | ✓ | ✓ |
| [FiftyOneImageDetectionDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#fiftyoneimagedetectiondataset) | ✓ | ✓ | ✓ |
| [COCODetectionDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#cocodetectiondataset) | ✓ | ✓ | ✓ |
| [VOCDetectionDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#vocdetectiondataset) | ✓ | ✓ | ✓ |
| [KITTIDetectionDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#kittidetectiondataset) | ✓ | ✓ | ✓ |
| [YOLODataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#yolodataset) | ✓ | ✓ | ✓ |
| [TFObjectDetectionDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#tfobjectdetectiondataset) | ✓ | ✓ | ✓ |
| [CVATImageDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#cvatimagedataset) | ✓ | ✓ | ✓ |
| [CVATVideoDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#cvatvideodataset) | ✓ | ✓ | ✓ |
| [FiftyOneImageLabelsDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#fiftyoneimagelabelsdataset) | ✓ | ✓ | ✓ |
| [FiftyOneVideoLabelsDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#fiftyonevideolabelsdataset) | ✓ | ✓ | ✓ |
| [BDDDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#bdddataset) | ✓ | ✓ | ✓ |

In addition, you can define your own [custom dataset types](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#custom-formats) to read/write datasets in your own formats.

The usage of the `fiftyone convert` command is as follows:

```
[1]:

```

```
fiftyone convert -h

```

```
usage: fiftyone convert [-h] --input-type INPUT_TYPE --output-type OUTPUT_TYPE
                        [--input-dir INPUT_DIR]
                        [--input-kwargs KEY=VAL [KEY=VAL ...]]
                        [--output-dir OUTPUT_DIR]
                        [--output-kwargs KEY=VAL [KEY=VAL ...]] [-o]

Convert datasets on disk between supported formats.

    Examples::

        # Convert an image classification directory tree to TFRecords format
        fiftyone convert \
            --input-dir /path/to/image-classification-directory-tree \
            --input-type fiftyone.types.ImageClassificationDirectoryTree \
            --output-dir /path/for/tf-image-classification-dataset \
            --output-type fiftyone.types.TFImageClassificationDataset

        # Convert a COCO detection dataset to CVAT image format
        fiftyone convert \
            --input-dir /path/to/coco-detection-dataset \
            --input-type fiftyone.types.COCODetectionDataset \
            --output-dir /path/for/cvat-image-dataset \
            --output-type fiftyone.types.CVATImageDataset

        # Perform a customized conversion via optional kwargs
        fiftyone convert \
            --input-dir /path/to/coco-detection-dataset \
            --input-type fiftyone.types.COCODetectionDataset \
            --input-kwargs max_samples=100 shuffle=True \
            --output-dir /path/for/cvat-image-dataset \
            --output-type fiftyone.types.TFObjectDetectionDataset \
            --output-kwargs force_rgb=True \
            --overwrite

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
                        the directory containing the dataset
  --input-kwargs KEY=VAL [KEY=VAL ...]
                        additional keyword arguments for `fiftyone.utils.data.convert_dataset(..., input_kwargs=)`
  --output-dir OUTPUT_DIR
                        the directory to which to write the output dataset
  --output-kwargs KEY=VAL [KEY=VAL ...]
                        additional keyword arguments for `fiftyone.utils.data.convert_dataset(..., output_kwargs=)`
  -o, --overwrite       whether to overwrite an existing output directory

required arguments:
  --input-type INPUT_TYPE
                        the fiftyone.types.Dataset type of the input dataset
  --output-type OUTPUT_TYPE
                        the fiftyone.types.Dataset type to output

```

## Convert CIFAR-10 dataset [¶](\#Convert-CIFAR-10-dataset "Permalink to this headline")

When you downloaded the test split of the CIFAR-10 dataset above, it was written to disk as a dataset in [fiftyone.types.FiftyOneImageClassificationDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#fiftyoneimageclassificationdataset) format.

You can verify this by printing information about the downloaded dataset:

```
[6]:

```

```
fiftyone zoo datasets info cifar10

```

```
***** Dataset description *****
The CIFAR-10 dataset consists of 60000 32 x 32 color images in 10
    classes, with 6000 images per class. There are 50000 training images and
    10000 test images.

    Dataset size:
        132.40 MiB

    Source:
        https://www.cs.toronto.edu/~kriz/cifar.html

***** Supported splits *****
test, train

***** Dataset location *****
~/fiftyone/cifar10

***** Dataset info *****
{
    "name": "cifar10",
    "zoo_dataset": "fiftyone.zoo.datasets.torch.CIFAR10Dataset",
    "dataset_type": "fiftyone.types.dataset_types.FiftyOneImageClassificationDataset",
    "num_samples": 10000,
    "downloaded_splits": {
        "test": {
            "split": "test",
            "num_samples": 10000
        }
    },
    "classes": [\
        "airplane",\
        "automobile",\
        "bird",\
        "cat",\
        "deer",\
        "dog",\
        "frog",\
        "horse",\
        "ship",\
        "truck"\
    ]
}

```

The snippet below uses `fiftyone convert` to convert the test split of the CIFAR-10 dataset to [fiftyone.types.ImageClassificationDirectoryTree](https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#imageclassificationdirectorytree) format, which stores classification datasets on disk in a directory tree structure with images organized per-class:

```
<dataset_dir>
├── <classA>/
│   ├── <image1>.<ext>
│   ├── <image2>.<ext>
│   └── ...
├── <classB>/
│   ├── <image1>.<ext>
│   ├── <image2>.<ext>
│   └── ...
└── ...

```

```
[7]:

```

```
INPUT_DIR=$(fiftyone zoo datasets find cifar10 --split test)
OUTPUT_DIR=/tmp/fiftyone/cifar10-dir-tree

fiftyone convert \
    --input-dir ${INPUT_DIR} --input-type fiftyone.types.FiftyOneImageClassificationDataset \
    --output-dir ${OUTPUT_DIR} --output-type fiftyone.types.ImageClassificationDirectoryTree

```

```
Loading dataset from '~/fiftyone/cifar10/test'
Input format 'fiftyone.types.dataset_types.FiftyOneImageClassificationDataset'
 100% |███| 10000/10000 [4.2s elapsed, 0s remaining, 2.4K samples/s]
Import complete
Exporting dataset to '/tmp/fiftyone/cifar10-dir-tree'
Export format 'fiftyone.types.dataset_types.ImageClassificationDirectoryTree'
 100% |███| 10000/10000 [6.2s elapsed, 0s remaining, 1.7K samples/s]
Export complete

```

Let’s verify that the conversion happened as expected:

```
[8]:

```

```
ls -lah /tmp/fiftyone/cifar10-dir-tree/

```

```
total 0
drwxr-xr-x    12 voxel51  wheel   384B Jul 14 11:08 .
drwxr-xr-x     3 voxel51  wheel    96B Jul 14 11:08 ..
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 airplane
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 automobile
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 bird
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 cat
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 deer
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 dog
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 frog
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 horse
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 ship
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 truck

```

```
[9]:

```

```
ls -lah /tmp/fiftyone/cifar10-dir-tree/airplane/ | head

```

```
total 8000
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 11:08 .
drwxr-xr-x    12 voxel51  wheel   384B Jul 14 11:08 ..
-rw-r--r--     1 voxel51  wheel   1.2K Jul 14 11:23 000004.jpg
-rw-r--r--     1 voxel51  wheel   1.1K Jul 14 11:23 000011.jpg
-rw-r--r--     1 voxel51  wheel   1.1K Jul 14 11:23 000022.jpg
-rw-r--r--     1 voxel51  wheel   1.3K Jul 14 11:23 000028.jpg
-rw-r--r--     1 voxel51  wheel   1.2K Jul 14 11:23 000045.jpg
-rw-r--r--     1 voxel51  wheel   1.2K Jul 14 11:23 000053.jpg
-rw-r--r--     1 voxel51  wheel   1.3K Jul 14 11:23 000075.jpg

```

Now let’s convert the classification directory tree to [TFRecords](https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#tfimageclassificationdataset) format!

```
[10]:

```

```
INPUT_DIR=/tmp/fiftyone/cifar10-dir-tree
OUTPUT_DIR=/tmp/fiftyone/cifar10-tfrecords

fiftyone convert \
    --input-dir ${INPUT_DIR} --input-type fiftyone.types.ImageClassificationDirectoryTree \
    --output-dir ${OUTPUT_DIR} --output-type fiftyone.types.TFImageClassificationDataset

```

```
Loading dataset from '/tmp/fiftyone/cifar10-dir-tree'
Input format 'fiftyone.types.dataset_types.ImageClassificationDirectoryTree'
 100% |███| 10000/10000 [4.0s elapsed, 0s remaining, 2.5K samples/s]
Import complete
Exporting dataset to '/tmp/fiftyone/cifar10-tfrecords'
Export format 'fiftyone.types.dataset_types.TFImageClassificationDataset'
   0% ||--|     1/10000 [23.2ms elapsed, 3.9m remaining, 43.2 samples/s] 2020-07-14 11:24:15.187387: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-14 11:24:15.201384: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f83df428f60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-14 11:24:15.201405: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
 100% |███| 10000/10000 [8.2s elapsed, 0s remaining, 1.3K samples/s]
Export complete

```

Let’s verify that the conversion happened as expected:

```
[11]:

```

```
ls -lah /tmp/fiftyone/cifar10-tfrecords

```

```
total 29696
drwxr-xr-x  3 voxel51  wheel    96B Jul 14 11:24 .
drwxr-xr-x  4 voxel51  wheel   128B Jul 14 11:24 ..
-rw-r--r--  1 voxel51  wheel    14M Jul 14 11:24 tf.records

```

## Convert KITTI dataset [¶](\#Convert-KITTI-dataset "Permalink to this headline")

When you downloaded the validation split of the KITTI dataset above, it was written to disk as a dataset in [fiftyone.types.FiftyOneImageDetectionDataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#fiftyoneimagedetectiondataset) format.

You can verify this by printing information about the downloaded dataset:

```
[12]:

```

```
fiftyone zoo datasets info kitti

```

```
***** Dataset description *****
KITTI contains a suite of vision tasks built using an autonomous
    driving platform.

    The full benchmark contains many tasks such as stereo, optical flow, visual
    odometry, etc. This dataset contains the object detection dataset,
    including the monocular images and bounding boxes. The dataset contains
    7481 training images annotated with 3D bounding boxes. A full description
    of the annotations can be found in the README of the object development kit
    on the KITTI homepage.

    Dataset size:
        5.27 GiB

    Source:
        http://www.cvlibs.net/datasets/kitti

***** Supported splits *****
test, train, validation

***** Dataset location *****
~/fiftyone/kitti

***** Dataset info *****
{
    "name": "kitti",
    "zoo_dataset": "fiftyone.zoo.datasets.tf.KITTIDataset",
    "dataset_type": "fiftyone.types.dataset_types.FiftyOneImageDetectionDataset",
    "num_samples": 423,
    "downloaded_splits": {
        "validation": {
            "split": "validation",
            "num_samples": 423
        }
    },
    "classes": [\
        "Car",\
        "Van",\
        "Truck",\
        "Pedestrian",\
        "Person_sitting",\
        "Cyclist",\
        "Tram",\
        "Misc"\
    ]
}

```

The snippet below uses `fiftyone convert` to convert the test split of the CIFAR-10 dataset to [fiftyone.types.COCODetectionDataset](https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#cocodetectiondataset) format, which writes the dataset to disk with annotations in [COCO format](https://cocodataset.org/#format-data).

```
[13]:

```

```
INPUT_DIR=$(fiftyone zoo datasets find kitti --split validation)
OUTPUT_DIR=/tmp/fiftyone/kitti-coco

fiftyone convert \
    --input-dir ${INPUT_DIR} --input-type fiftyone.types.FiftyOneImageDetectionDataset \
    --output-dir ${OUTPUT_DIR} --output-type fiftyone.types.COCODetectionDataset

```

```
Loading dataset from '~/fiftyone/kitti/validation'
Input format 'fiftyone.types.dataset_types.FiftyOneImageDetectionDataset'
 100% |███████| 423/423 [1.2s elapsed, 0s remaining, 351.0 samples/s]
Import complete
Exporting dataset to '/tmp/fiftyone/kitti-coco'
Export format 'fiftyone.types.dataset_types.COCODetectionDataset'
 100% |███████| 423/423 [4.4s elapsed, 0s remaining, 96.1 samples/s]
Export complete

```

Let’s verify that the conversion happened as expected:

```
[14]:

```

```
ls -lah /tmp/fiftyone/kitti-coco/

```

```
total 880
drwxr-xr-x    4 voxel51  wheel   128B Jul 14 11:24 .
drwxr-xr-x    5 voxel51  wheel   160B Jul 14 11:24 ..
drwxr-xr-x  425 voxel51  wheel    13K Jul 14 11:24 data
-rw-r--r--    1 voxel51  wheel   437K Jul 14 11:24 labels.json

```

```
[15]:

```

```
ls -lah /tmp/fiftyone/kitti-coco/data | head

```

```
total 171008
drwxr-xr-x  425 voxel51  wheel    13K Jul 14 11:24 .
drwxr-xr-x    4 voxel51  wheel   128B Jul 14 11:24 ..
-rw-r--r--    1 voxel51  wheel   195K Jul 14 11:24 000001.jpg
-rw-r--r--    1 voxel51  wheel   191K Jul 14 11:24 000002.jpg
-rw-r--r--    1 voxel51  wheel   167K Jul 14 11:24 000003.jpg
-rw-r--r--    1 voxel51  wheel   196K Jul 14 11:24 000004.jpg
-rw-r--r--    1 voxel51  wheel   224K Jul 14 11:24 000005.jpg
-rw-r--r--    1 voxel51  wheel   195K Jul 14 11:24 000006.jpg
-rw-r--r--    1 voxel51  wheel   177K Jul 14 11:24 000007.jpg

```

```
[19]:

```

```
cat /tmp/fiftyone/kitti-coco/labels.json | python -m json.tool 2> /dev/null | head -20
echo "..."
cat /tmp/fiftyone/kitti-coco/labels.json | python -m json.tool 2> /dev/null | tail -20

```

```
{
    "info": {
        "year": "",
        "version": "",
        "description": "Exported from FiftyOne",
        "contributor": "",
        "url": "https://voxel51.com/fiftyone",
        "date_created": "2020-07-14T11:24:40"
    },
    "licenses": [],
    "categories": [\
        {\
            "id": 0,\
            "name": "Car",\
            "supercategory": "none"\
        },\
        {\
            "id": 1,\
            "name": "Cyclist",\
            "supercategory": "none"\
...\
            "area": 4545.8,\
            "segmentation": null,\
            "iscrowd": 0\
        },\
        {\
            "id": 3196,\
            "image_id": 422,\
            "category_id": 3,\
            "bbox": [\
                367.2,\
                107.3,\
                36.2,\
                105.2\
            ],\
            "area": 3808.2,\
            "segmentation": null,\
            "iscrowd": 0\
        }\
    ]
}

```

Now let’s convert from COCO format to [CVAT Image format](https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#cvatimageformat) format!

```
[20]:

```

```
INPUT_DIR=/tmp/fiftyone/kitti-coco
OUTPUT_DIR=/tmp/fiftyone/kitti-cvat

fiftyone convert \
    --input-dir ${INPUT_DIR} --input-type fiftyone.types.COCODetectionDataset \
    --output-dir ${OUTPUT_DIR} --output-type fiftyone.types.CVATImageDataset

```

```
Loading dataset from '/tmp/fiftyone/kitti-coco'
Input format 'fiftyone.types.dataset_types.COCODetectionDataset'
 100% |███████| 423/423 [2.0s elapsed, 0s remaining, 206.4 samples/s]
Import complete
Exporting dataset to '/tmp/fiftyone/kitti-cvat'
Export format 'fiftyone.types.dataset_types.CVATImageDataset'
 100% |███████| 423/423 [1.3s elapsed, 0s remaining, 323.7 samples/s]
Export complete

```

Let’s verify that the conversion happened as expected:

```
[21]:

```

```
ls -lah /tmp/fiftyone/kitti-cvat

```

```
total 584
drwxr-xr-x    4 voxel51  wheel   128B Jul 14 11:25 .
drwxr-xr-x    6 voxel51  wheel   192B Jul 14 11:25 ..
drwxr-xr-x  425 voxel51  wheel    13K Jul 14 11:25 data
-rw-r--r--    1 voxel51  wheel   289K Jul 14 11:25 labels.xml

```

```
[22]:

```

```
cat /tmp/fiftyone/kitti-cvat/labels.xml | head -20
echo "..."
cat /tmp/fiftyone/kitti-cvat/labels.xml | tail -20

```

```
<?xml version="1.0" encoding="utf-8"?>
<annotations>
    <version>1.1</version>
    <meta>
        <task>
            <size>423</size>
            <mode>annotation</mode>
            <labels>
                <label>
                    <name>Car</name>
                    <attributes>
                    </attributes>
                </label>
                <label>
                    <name>Cyclist</name>
                    <attributes>
                    </attributes>
                </label>
                <label>
                    <name>Misc</name>
...
        <box label="Pedestrian" xtl="360" ytl="116" xbr="402" ybr="212">
        </box>
        <box label="Pedestrian" xtl="396" ytl="120" xbr="430" ybr="212">
        </box>
        <box label="Pedestrian" xtl="413" ytl="112" xbr="483" ybr="212">
        </box>
        <box label="Pedestrian" xtl="585" ytl="80" xbr="646" ybr="215">
        </box>
        <box label="Pedestrian" xtl="635" ytl="94" xbr="688" ybr="212">
        </box>
        <box label="Pedestrian" xtl="422" ytl="85" xbr="469" ybr="210">
        </box>
        <box label="Pedestrian" xtl="457" ytl="93" xbr="520" ybr="213">
        </box>
        <box label="Pedestrian" xtl="505" ytl="101" xbr="548" ybr="206">
        </box>
        <box label="Pedestrian" xtl="367" ytl="107" xbr="403" ybr="212">
        </box>
    </image>
</annotations>

```

## Cleanup [¶](\#Cleanup "Permalink to this headline")

You can cleanup the files generated by this recipe by running the command below:

```
[23]:

```

```
rm -rf /tmp/fiftyone

```

- Convert Dataset Formats
  - [Setup](#Setup)
  - [Download datasets](#Download-datasets)
  - [The fiftyone convert command](#The-fiftyone-convert-command)
  - [Convert CIFAR-10 dataset](#Convert-CIFAR-10-dataset)
  - [Convert KITTI dataset](#Convert-KITTI-dataset)
  - [Cleanup](#Cleanup)