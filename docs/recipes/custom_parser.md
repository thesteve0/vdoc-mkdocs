# Writing Custom Sample Parsers [¶](\#Writing-Custom-Sample-Parsers "Permalink to this headline")

This recipe demonstrates how to write a [custom SampleParser](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/samples.html#writing-a-custom-sampleparser) and use it to add samples in your custom format to a FiftyOne Dataset.

## Setup [¶](\#Setup "Permalink to this headline")

If you haven’t already, install FiftyOne:

```python
[ ]:

```

```python
!pip install fiftyone

```

In this receipe we’ll use the [TorchVision Datasets](https://pytorch.org/vision/stable/datasets.html) library to download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) to use as sample data to feed our custom parser.

You can install the necessary packages, if necessary, as follows:

```python
[ ]:

```

```python
!pip install torch torchvision

```

## Writing a SampleParser [¶](\#Writing-a-SampleParser "Permalink to this headline")

FiftyOne provides a [SampleParser](https://voxel51.com/docs/fiftyone/api/fiftyone.utils.data.html#fiftyone.utils.data.parsers.SampleParser) interface that defines how it parses provided samples when methods such as [Dataset.add\_labeled\_images()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.html#fiftyone.core.dataset.Dataset.add_labeled_images) and
[Dataset.ingest\_labeled\_images()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.html#fiftyone.core.dataset.Dataset.ingest_labeled_images) are used.

`SampleParser` itself is an abstract interface; the concrete interface that you should implement is determined by the type of samples that you are importing. See [writing a custom SampleParser](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/samples.html#writing-a-custom-sampleparser) for full details.

In this recipe, we’ll write a custom [LabeledImageSampleParser](https://voxel51.com/docs/fiftyone/api/fiftyone.utils.data.html#fiftyone.utils.data.parsers.LabeledImageSampleParser) that can parse labeled images from a [PyTorch Dataset](https://pytorch.org/docs/stable/data.html).

Here’s the complete definition of the `SampleParser`:

```python
[1]:

```

```python
import fiftyone as fo
import fiftyone.utils.data as foud

class PyTorchClassificationDatasetSampleParser(foud.LabeledImageSampleParser):
    """Parser for image classification samples loaded from a PyTorch dataset.

    This parser can parse samples from a ``torch.utils.data.DataLoader`` that
    emits ``(img_tensor, target)`` tuples, where::

        - `img_tensor`: is a PyTorch Tensor containing the image
        - `target`: the integer index of the target class

    Args:
        classes: the list of class label strings
    """

    def __init__(self, classes):
        self.classes = classes

    @property
    def has_image_path(self):
        """Whether this parser produces paths to images on disk for samples
        that it parses.
        """
        return False

    @property
    def has_image_metadata(self):
        """Whether this parser produces
        :class:`fiftyone.core.metadata.ImageMetadata` instances for samples
        that it parses.
        """
        return False

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) returned by this
        parser.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            parser is guaranteed to return labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the parser can produce a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the parser will return label dictionaries with keys
            and value-types specified by this dictionary. Not all keys need be
            present in the imported labels
        -   ``None``. In this case, the parser makes no guarantees about the
            labels that it may return
        """
        return fo.Classification

    def get_image(self):
        """Returns the image from the current sample.

        Returns:
            a numpy image
        """
        img_tensor = self.current_sample[0]
        return img_tensor.cpu().numpy()

    def get_label(self):
        """Returns the label for the current sample.

        Returns:
            a :class:`fiftyone.core.labels.Label` instance, or a dictionary
            mapping field names to :class:`fiftyone.core.labels.Label`
            instances, or ``None`` if the sample is unlabeled
        """
        target = self.current_sample[1]
        return fo.Classification(label=self.classes[int(target)])

```

Note that `PyTorchClassificationDatasetSampleParser` specifies `has_image_path == False` and `has_image_metadata == False`, because the PyTorch dataset directly provides the in-memory image, not its path on disk.

## Ingesting samples into a dataset [¶](\#Ingesting-samples-into-a-dataset "Permalink to this headline")

In order to use `PyTorchClassificationDatasetSampleParser`, we need a PyTorch Dataset from which to feed it samples.

Let’s use the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) from the [TorchVision Datasets](https://pytorch.org/docs/stable/torchvision/datasets.html) library:

```python
[2]:

```

```python
import torch
import torchvision

# Downloads the test split of the CIFAR-10 dataset and prepares it for loading
# in a DataLoader
dataset = torchvision.datasets.CIFAR10(
    "/tmp/fiftyone/custom-parser/pytorch",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
classes = dataset.classes
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

```

```python
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to /tmp/fiftyone/custom-parser/pytorch/cifar-10-python.tar.gz

```

```python
Extracting /tmp/fiftyone/custom-parser/pytorch/cifar-10-python.tar.gz to /tmp/fiftyone/custom-parser/pytorch

```

Now we can load the samples into the dataset. Since our custom sample parser declares `has_image_path == False`, we must use the [Dataset.ingest\_labeled\_images()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.html#fiftyone.core.dataset.Dataset.ingest_labeled_images) method to load the samples into a FiftyOne dataset, which will write the individual images to disk as they are ingested so that FiftyOne can access them.

```python
[3]:

```

```python
dataset = fo.Dataset("cifar10-samples")

sample_parser = PyTorchClassificationDatasetSampleParser(classes)

# The directory to use to store the individual images on disk
dataset_dir = "/tmp/fiftyone/custom-parser/fiftyone"

# Ingest the samples from the data loader
dataset.ingest_labeled_images(data_loader, sample_parser, dataset_dir=dataset_dir)

print("Loaded %d samples" % len(dataset))

```

```python
 100% |███| 10000/10000 [6.7s elapsed, 0s remaining, 1.5K samples/s]
Loaded 10000 samples

```

Let’s inspect the contents of the dataset to verify that the samples were loaded as expected:

```python
[4]:

```

```python
# Print summary information about the dataset
print(dataset)

```

```python
Name:           cifar10-samples
Persistent:     False
Num samples:    10000
Tags:           []
Sample fields:
    filepath:     fiftyone.core.fields.StringField
    tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth: fiftyone.core.fields.StringField

```

```python
[5]:

```

```python
# Print a few samples from the dataset
print(dataset.head())

```

```python
<Sample: {
    'dataset_name': 'cifar10-samples',
    'id': '5f15aeab6d4e59654468a14e',
    'filepath': '/tmp/fiftyone/custom-parser/fiftyone/000001.jpg',
    'tags': BaseList([]),
    'metadata': None,
    'ground_truth': 'cat',
}>
<Sample: {
    'dataset_name': 'cifar10-samples',
    'id': '5f15aeab6d4e59654468a14f',
    'filepath': '/tmp/fiftyone/custom-parser/fiftyone/000002.jpg',
    'tags': BaseList([]),
    'metadata': None,
    'ground_truth': 'ship',
}>
<Sample: {
    'dataset_name': 'cifar10-samples',
    'id': '5f15aeab6d4e59654468a150',
    'filepath': '/tmp/fiftyone/custom-parser/fiftyone/000003.jpg',
    'tags': BaseList([]),
    'metadata': None,
    'ground_truth': 'ship',
}>

```

We can also verify that the ingested images were written to disk as expected:

```python
[27]:

```

```python
!ls -lah /tmp/fiftyone/custom-parser/fiftyone | head -n 10

```

```python
total 0
drwxr-xr-x  10002 voxel51  wheel   313K Jul 20 10:34 .
drwxr-xr-x      4 voxel51  wheel   128B Jul 20 10:34 ..
-rw-r--r--      1 voxel51  wheel     0B Jul 20 10:34 000001.jpg
-rw-r--r--      1 voxel51  wheel     0B Jul 20 10:34 000002.jpg
-rw-r--r--      1 voxel51  wheel     0B Jul 20 10:34 000003.jpg
-rw-r--r--      1 voxel51  wheel     0B Jul 20 10:34 000004.jpg
-rw-r--r--      1 voxel51  wheel     0B Jul 20 10:34 000005.jpg
-rw-r--r--      1 voxel51  wheel     0B Jul 20 10:34 000006.jpg
-rw-r--r--      1 voxel51  wheel     0B Jul 20 10:34 000007.jpg

```

## Adding samples to a dataset [¶](\#Adding-samples-to-a-dataset "Permalink to this headline")

If our `LabeledImageSampleParser` declared `has_image_path == True`, then we could use [Dataset.add\_labeled\_images()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.html#fiftyone.core.dataset.Dataset.add_labeled_images) to add samples to FiftyOne datasets without creating a copy of the source images on disk.

However, our sample parser does not provide image paths, so an informative error message is raised if we try to use it in an unsupported way:

```python
[6]:

```

```python
dataset = fo.Dataset()

sample_parser = PyTorchClassificationDatasetSampleParser(classes)

# Won't work because our SampleParser does not provide paths to its source images on disk
dataset.add_labeled_images(data_loader, sample_parser)

```

```python
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-6-a3d739e371af> in <module>
      4
      5 # Won't work because our SampleParser does not provide paths to its source images on disk
----> 6 dataset.add_labeled_images(data_loader, sample_parser)

~/dev/fiftyone/fiftyone/core/dataset.py in add_labeled_images(self, samples, sample_parser, label_field, tags, expand_schema)
    729         if not sample_parser.has_image_path:
    730             raise ValueError(
--> 731                 "Sample parser must have `has_image_path == True` to add its "
    732                 "samples to the dataset"
    733             )

ValueError: Sample parser must have `has_image_path == True` to add its samples to the dataset

```

## Cleanup [¶](\#Cleanup "Permalink to this headline")

You can cleanup the files generated by this recipe by running:

```python
[7]:

```

```python
!rm -rf /tmp/fiftyone

```

