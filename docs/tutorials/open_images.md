# Downloading and Evaluating Open Images [¶](\#Downloading-and-Evaluating-Open-Images "Permalink to this headline")

Downloading Google’s [Open Images dataset](https://storage.googleapis.com/openimages/web/download.html) is now easier than ever with the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/index.html#dataset-zoo-open-images-v7)! You can load all three splits of Open Images V7, including image-level labels, detections, segmentations, visual relationships, and point labels.

FiftyOne also natively supports [Open Images-style evaluation](https://voxel51.com/docs/fiftyone/user_guide/evaluation.html#open-images-style-evaluation), so you can easily evaluate your object detection models and explore the results directly in the library.

This walkthrough covers:

- Downloading [Open Images](https://storage.googleapis.com/openimages/web/index.html) from the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/index.html)

- Computing predictions using a model from the [FiftyOne Model Zoo](https://voxel51.com/docs/fiftyone/user_guide/model_zoo/index.html)

- Performing [Open Images-style evaluation](https://voxel51.com/docs/fiftyone/user_guide/evaluation.html#open-images-style-evaluation) in FiftyOne to evaluate a model and compute its mAP

- Exploring the dataset and [evaluation results](https://voxel51.com/docs/fiftyone/user_guide/evaluation.html)

- [Visualizing embeddings](https://voxel51.com/docs/fiftyone/user_guide/brain.html#visualizing-embeddings) through [interactive plots](https://voxel51.com/docs/fiftyone/user_guide/plots.html)


**So, what’s the takeaway?**

Starting a new ML project takes data and time, and the datasets in the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/index.html) can help jump start the development process.

Open Images in particular is one of the largest publicly available datasets for object detections, classification, segmentation, and more. Additionally, with [Open Images evaluation](https://voxel51.com/docs/fiftyone/user_guide/evaluation.html#open-images-style-evaluation) available natively in FiftyOne, you can quickly evaluate your models and compute mAP and PR curves.

While metrics like mAP are often used to compare models, the best way to improve your model’s performance isn’t to look at aggregate metrics but instead to get hands-on with your evaluation and visualize how your model performs on individual samples. All of this is made easy with FiftyOne!

## Setup [¶](\#Setup "Permalink to this headline")

If you haven’t already, install FiftyOne:

```python
[1]:

```

```python
!pip install fiftyone

```

In this tutorial, we’ll use some [TensorFlow models](https://github.com/tensorflow/models) and [PyTorch](https://pytorch.org/vision/stable/index.html) to generate predictions and embeddings, and we’ll use the [UMAP method](https://github.com/lmcinnes/umap) to reduce the dimensionality of embeddings, so we need to install the corresponding packages:

```python
[2]:

```

```python
!pip install tensorflow torch torchvision umap-learn

```

This tutorial also includes some of FiftyOne’s [interactive plotting capabilities](https://voxel51.com/docs/fiftyone/user_guide/plots.html).

The recommended way to work with FiftyOne’s interactive plots is in [Jupyter notebooks](https://jupyter.org/) or [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/). In these environments, you can leverage the full power of plots by [attaching them to the FiftyOne App](https://voxel51.com/docs/fiftyone/user_guide/plots.html#attaching-plots) and bidirectionally interacting with the plots and the App to identify interesting subsets of your data.

To use interactive plots in Jupyter notebooks, ensure that you have the `ipywidgets` package installed:

```python
[3]:

```

```python
!pip install 'ipywidgets>=8,<9'

```

If you’re working in JupyterLab, refer to [these instructions](https://voxel51.com/docs/fiftyone/user_guide/plots.html#working-in-notebooks) to get setup.

Support for interactive plots in non-notebook contexts and Google Colab is coming soon! In the meantime, you can still use FiftyOne’s plotting features in those environments, but you must manually call plot.show() to update the state of a plot to match the state of a connected session, and any callbacks that would normally be triggered in response to interacting with a plot will not be triggered.

## Loading Open Images [¶](\#Loading-Open-Images "Permalink to this headline")

In this section, we’ll load various subsets of Open Images from the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/index.html) and visualize them using FiftyOne.

Let’s start by downloading a small sample of 100 randomly chosen images + annotations:

```python
[4]:

```

```python
import fiftyone as fo
import fiftyone.zoo as foz

```

```python
[5]:

```

```python
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    max_samples=100,
    seed=51,
    shuffle=True,
)

```

Now let’s launch the [FiftyOne App](https://voxel51.com/docs/fiftyone/user_guide/app.html) so we can explore the [dataset](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#datasets) we just downloaded.

```python
[6]:

```

```python
session = fo.launch_app(dataset.view())

```

```python
Connected to FiftyOne on port 5151 at localhost.
If you are not connecting to a remote session, you may need to start a new session and specify a port

```

Activate

![](<Base64-Image-Removed>)

Loading Open Images with FiftyOne also automatically stores relevant [labels](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#labels) and [metadata](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#metadata) like classes, attributes, and a class hierarchy that is used for evaluation in the dataset’s `info` dictionary:

```python
[7]:

```

```python
print(dataset.info.keys())

```

```python
dict_keys(['hierarchy', 'attributes_map', 'attributes', 'segmentation_classes', 'point_classes', 'classes_map'])

```

When loading Open Images from the dataset zoo, there are a [variety of available parameters](https://voxel51.com/docs/fiftyone/api/fiftyone.zoo.datasets.base.html#fiftyone.zoo.datasets.base.OpenImagesV7Dataset) that you can pass to `load_zoo_dataset()` to specify a subset of the images and/or label types to download:

- `label_types` \- a list of label types to load. The supported values are ( `"detections", "classifications", "points", "segmentations", "relationships"`) for Open Images V7. Open Images v6 is the same except that it does not contain point labels. By default, all available labels types will be loaded. Specifying `[]` will load only the images

- `classes` \- a list of classes of interest. If specified, only samples with at least one object, segmentation, or image-level label in the specified classes will be downloaded

- `attrs` \- a list of attributes of interest. If specified, only download samples if they contain at least one attribute in `attrs` or one class in `classes` (only applicable when `label_types` contains `"relationships"`)

- `load_hierarchy` \- whether to load the class hierarchy into `dataset.info["hierarchy"]`

- `image_ids` \- an array of specific image IDs to download

- `image_ids_file` \- a path to a `.txt`, `.csv`, or `.json` file containing image IDs to download


In addition, [like all other zoo datasets](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html), you can specify:

- `max_samples` \- the maximum number of samples to load

- `shuffle` \- whether to randomly chose which samples to load if `max_samples` is given

- `seed` \- a random seed to use when shuffling


Let’s use some of these parameters to download a 100 sample subset of Open Images containing segmentations and image-level labels for the classes “Burrito”, “Cheese”, and “Popcorn”.

```python
[8]:

```

```python
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["segmentations", "classifications"],
    classes = ["Burrito", "Cheese", "Popcorn"],
    max_samples=100,
    seed=51,
    shuffle=True,
    dataset_name="open-images-food",
)

```

```python
Downloading split 'validation' to 'datasets/open-images-v7/validation' if necessary
Only found 83 (<100) samples matching your requirements
Necessary images already downloaded
Existing download of split 'validation' is sufficient
Loading existing dataset 'open-images-food'. To reload from disk, either delete the existing dataset or provide a custom `dataset_name` to use

```

```python
[9]:

```

```python
session.view = dataset.view()

```

Activate

![](<Base64-Image-Removed>)

```python
[10]:

```

```python
session.freeze() # screenshots App for sharing

```

We can do the same for visual relationships. For example, we can download only samples that contain a relationship with the “Wooden” attribute.

```python
[ ]:

```

```python
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["relationships"],
    attrs=["Wooden"],
    max_samples=100,
    seed=51,
    shuffle=True,
    dataset_name="open-images-relationships",
)

```

You can visualize relationships in the App by clicking on a sample to open the [App’s expanded view](https://voxel51.com/docs/fiftyone/user_guide/app.html#viewing-a-sample). From there, you can hover over objects to see their attributes in a tooltip.

Alternatively, you can use the settings menu in the lower-right corner of the media player to set `show_attributes` to True to make attributes appear as persistent boxes (as shown below). This can also be achieved programmatically by [configuring the App](https://voxel51.com/docs/fiftyone/user_guide/config.html#configuring-the-app):

```python
[12]:

```

```python
# Launch a new App instance with a customized config
app_config = fo.AppConfig()
app_config.show_attributes = True

session = fo.launch_app(dataset, config=app_config)

```

Activate

![](<Base64-Image-Removed>)