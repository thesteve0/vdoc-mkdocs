# Writing Custom Dataset Exporters [¶](\#Writing-Custom-Dataset-Exporters "Permalink to this headline")

This recipe demonstrates how to write a [custom DatasetExporter](https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#custom-formats) and use it to export a FiftyOne dataset to disk in your custom format.

## Setup [¶](\#Setup "Permalink to this headline")

If you haven’t already, install FiftyOne:

```python
[ ]:

```

```python
!pip install fiftyone

```

In this recipe we’ll use the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/zoo_datasets.html) to download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) to use as sample data to feed our custom exporter.

Behind the scenes, FiftyOne uses either the [TensorFlow Datasets](https://www.tensorflow.org/datasets) or [TorchVision Datasets](https://pytorch.org/vision/stable/datasets.html) libraries to wrangle the datasets, depending on which ML library you have installed.

You can, for example, install PyTorch as follows:

```python
[ ]:

```

```python
!pip install torch torchvision

```

## Writing a DatasetExporter [¶](\#Writing-a-DatasetExporter "Permalink to this headline")

FiftyOne provides a [DatasetExporter](https://voxel51.com/docs/fiftyone/api/fiftyone.utils.data.html#fiftyone.utils.data.exporters.DatasetExporter) interface that defines how it exports datasets to disk when methods such as [Dataset.export()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.html#fiftyone.core.dataset.Dataset.export) are used.

`DatasetExporter` itself is an abstract interface; the concrete interface that you should implement is determined by the type of dataset that you are exporting. See [writing a custom DatasetExporter](https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#custom-formats) for full details.

In this recipe, we’ll write a custom [LabeledImageDatasetExporter](https://voxel51.com/docs/fiftyone/api/fiftyone.utils.data.html#fiftyone.utils.data.exporters.LabeledImageDatasetExporter) that can export an image classification dataset to disk in the following format:

```python
<dataset_dir>/
    data/
        <filename1>.<ext>
        <filename2>.<ext>
        ...
    labels.csv

```

where `labels.csv` is a CSV file that contains the image metadata and associated labels in the following format:

```python
filepath,size_bytes,mime_type,width,height,num_channels,label
<filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
<filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
...

```

Here’s the complete definition of the `DatasetExporter`:

```python
[1]:

```

```python
import csv
import os

import fiftyone as fo
import fiftyone.utils.data as foud

class CSVImageClassificationDatasetExporter(foud.LabeledImageDatasetExporter):
    """Exporter for image classification datasets whose labels and image
    metadata are stored on disk in a CSV file.

    Datasets of this type are exported in the following format:

        <dataset_dir>/
            data/
                <filename1>.<ext>
                <filename2>.<ext>
                ...
            labels.csv

    where ``labels.csv`` is a CSV file in the following format::

        filepath,size_bytes,mime_type,width,height,num_channels,label
        <filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
        <filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
        ...

    Args:
        export_dir: the directory to write the export
    """

    def __init__(self, export_dir):
        super().__init__(export_dir=export_dir)
        self._data_dir = None
        self._labels_path = None
        self._labels = None
        self._image_exporter = None

    @property
    def requires_image_metadata(self):
        """Whether this exporter requires
        :class:`fiftyone.core.metadata.ImageMetadata` instances for each sample
        being exported.
        """
        return True

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) exported by this
        exporter.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            exporter directly exports labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the exporter can export a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the exporter can handle label dictionaries with
            value-types specified by this dictionary. Not all keys need be
            present in the exported label dicts
        -   ``None``. In this case, the exporter makes no guarantees about the
            labels that it can export
        """
        return fo.Classification

    def setup(self):
        """Performs any necessary setup before exporting the first sample in
        the dataset.

        This method is called when the exporter's context manager interface is
        entered, :func:`DatasetExporter.__enter__`.
        """
        self._data_dir = os.path.join(self.export_dir, "data")
        self._labels_path = os.path.join(self.export_dir, "labels.csv")
        self._labels = []

        # The `ImageExporter` utility class provides an `export()` method
        # that exports images to an output directory with automatic handling
        # of things like name conflicts
        self._image_exporter = foud.ImageExporter(
            True, export_path=self._data_dir, default_ext=".jpg",
        )
        self._image_exporter.setup()

    def export_sample(self, image_or_path, label, metadata=None):
        """Exports the given sample to the dataset.

        Args:
            image_or_path: an image or the path to the image on disk
            label: an instance of :meth:`label_cls`, or a dictionary mapping
                field names to :class:`fiftyone.core.labels.Label` instances,
                or ``None`` if the sample is unlabeled
            metadata (None): a :class:`fiftyone.core.metadata.ImageMetadata`
                instance for the sample. Only required when
                :meth:`requires_image_metadata` is ``True``
        """
        out_image_path, _ = self._image_exporter.export(image_or_path)

        if metadata is None:
            metadata = fo.ImageMetadata.build_for(image_or_path)

        self._labels.append((
            out_image_path,
            metadata.size_bytes,
            metadata.mime_type,
            metadata.width,
            metadata.height,
            metadata.num_channels,
            label.label,  # here, `label` is a `Classification` instance
        ))

    def close(self, *args):
        """Performs any necessary actions after the last sample has been
        exported.

        This method is called when the exporter's context manager interface is
        exited, :func:`DatasetExporter.__exit__`.

        Args:
            *args: the arguments to :func:`DatasetExporter.__exit__`
        """
        # Ensure the base output directory exists
        basedir = os.path.dirname(self._labels_path)
        if basedir and not os.path.isdir(basedir):
            os.makedirs(basedir)

        # Write the labels CSV file
        with open(self._labels_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow([\
                "filepath",\
                "size_bytes",\
                "mime_type",\
                "width",\
                "height",\
                "num_channels",\
                "label",\
            ])
            for row in self._labels:
                writer.writerow(row)

```

## Generating a sample dataset [¶](\#Generating-a-sample-dataset "Permalink to this headline")

In order to use `CSVImageClassificationDatasetExporter`, we need some labeled image samples to work with.

Let’s use some samples from the test split of CIFAR-10:

```python
[2]:

```

```python
import fiftyone.zoo as foz

num_samples = 1000

#
# Load `num_samples` from CIFAR-10
#
# This command will download the test split of CIFAR-10 from the web the first
# time it is executed, if necessary
#
cifar10_test = foz.load_zoo_dataset("cifar10", split="test")
samples = cifar10_test.limit(num_samples)

```

```python
Split 'test' already downloaded
Loading 'cifar10' split 'test'
 100% |███| 10000/10000 [4.4s elapsed, 0s remaining, 2.2K samples/s]

```

```python
[3]:

```

```python
# Print summary information about the samples
print(samples)

```

```python
Dataset:        cifar10-test
Num samples:    1000
Tags:           ['test']
Sample fields:
    filepath:     fiftyone.core.fields.StringField
    tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
Pipeline stages:
    1. Limit(limit=1000)

```

```python
[4]:

```

```python
# Print a sample
print(samples.first())

```

```python
<Sample: {
    'dataset_name': 'cifar10-test',
    'id': '5f0e6d7f503bf2b87254061c',
    'filepath': '~/fiftyone/cifar10/test/data/000001.jpg',
    'tags': BaseList(['test']),
    'metadata': None,
    'ground_truth': <Classification: {'label': 'cat', 'confidence': None, 'logits': None}>,
}>

```

## Exporting a dataset [¶](\#Exporting-a-dataset "Permalink to this headline")

With our samples and `DatasetExporter` in-hand, exporting the samples to disk in our custom format is as simple as follows:

```python
[5]:

```

```python
export_dir = "/tmp/fiftyone/custom-dataset-exporter"

# Export the dataset
print("Exporting %d samples to '%s'" % (len(samples), export_dir))
exporter = CSVImageClassificationDatasetExporter(export_dir)
samples.export(dataset_exporter=exporter)

```

```python
Exporting 1000 samples to '/tmp/fiftyone/custom-dataset-exporter'
 100% |█████| 1000/1000 [1.0s elapsed, 0s remaining, 1.0K samples/s]

```

Let’s inspect the contents of the exported dataset to verify that it was written in the correct format:

```python
[9]:

```

```python
!ls -lah /tmp/fiftyone/custom-dataset-exporter

```

```python
total 168
drwxr-xr-x     4 voxel51  wheel   128B Jul 14 22:46 .
drwxr-xr-x     3 voxel51  wheel    96B Jul 14 22:46 ..
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 22:46 data
-rw-r--r--     1 voxel51  wheel    83K Jul 14 22:46 labels.csv

```

```python
[10]:

```

```python
!ls -lah /tmp/fiftyone/custom-dataset-exporter/data | head -n 10

```

```python
total 8000
drwxr-xr-x  1002 voxel51  wheel    31K Jul 14 22:46 .
drwxr-xr-x     4 voxel51  wheel   128B Jul 14 22:46 ..
-rw-r--r--     1 voxel51  wheel   1.4K Jul 14 22:46 000001.jpg
-rw-r--r--     1 voxel51  wheel   1.3K Jul 14 22:46 000002.jpg
-rw-r--r--     1 voxel51  wheel   1.2K Jul 14 22:46 000003.jpg
-rw-r--r--     1 voxel51  wheel   1.2K Jul 14 22:46 000004.jpg
-rw-r--r--     1 voxel51  wheel   1.4K Jul 14 22:46 000005.jpg
-rw-r--r--     1 voxel51  wheel   1.3K Jul 14 22:46 000006.jpg
-rw-r--r--     1 voxel51  wheel   1.4K Jul 14 22:46 000007.jpg

```

```python
[11]:

```

```python
!head -n 10 /tmp/fiftyone/custom-dataset-exporter/labels.csv

```

```python
filepath,size_bytes,mime_type,width,height,num_channels,label
/tmp/fiftyone/custom-dataset-exporter/data/000001.jpg,1422,image/jpeg,32,32,3,cat
/tmp/fiftyone/custom-dataset-exporter/data/000002.jpg,1285,image/jpeg,32,32,3,ship
/tmp/fiftyone/custom-dataset-exporter/data/000003.jpg,1258,image/jpeg,32,32,3,ship
/tmp/fiftyone/custom-dataset-exporter/data/000004.jpg,1244,image/jpeg,32,32,3,airplane
/tmp/fiftyone/custom-dataset-exporter/data/000005.jpg,1388,image/jpeg,32,32,3,frog
/tmp/fiftyone/custom-dataset-exporter/data/000006.jpg,1311,image/jpeg,32,32,3,frog
/tmp/fiftyone/custom-dataset-exporter/data/000007.jpg,1412,image/jpeg,32,32,3,automobile
/tmp/fiftyone/custom-dataset-exporter/data/000008.jpg,1218,image/jpeg,32,32,3,frog
/tmp/fiftyone/custom-dataset-exporter/data/000009.jpg,1262,image/jpeg,32,32,3,cat

```

## Cleanup [¶](\#Cleanup "Permalink to this headline")

You can cleanup the files generated by this recipe by running:

```python
[12]:

```

```python
!rm -rf /tmp/fiftyone

```

