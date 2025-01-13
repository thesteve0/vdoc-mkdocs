# Writing Custom Dataset Importers [¶](\#Writing-Custom-Dataset-Importers "Permalink to this headline")

This recipe demonstrates how to write a [custom DatasetImporter](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#custom-formats) and use it to load a dataset from disk in your custom format into FiftyOne.

## Setup [¶](\#Setup "Permalink to this headline")

If you haven’t already, install FiftyOne:

```python
[ ]:

```

```python
!pip install fiftyone

```

In this recipe we’ll use the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/zoo_datasets.html) to download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) to use as sample data to feed our custom importer.

Behind the scenes, FiftyOne either uses the [TensorFlow Datasets](https://www.tensorflow.org/datasets) or [TorchVision Datasets](https://pytorch.org/vision/stable/datasets.html) libraries to wrangle the datasets, depending on which ML library you have installed.

You can, for example, install PyTorch as follows:

```python
[ ]:

```

```python
!pip install torch torchvision

```

## Writing a DatasetImporter [¶](\#Writing-a-DatasetImporter "Permalink to this headline")

FiftyOne provides a [DatasetImporter](https://voxel51.com/docs/fiftyone/api/fiftyone.utils.data.html#fiftyone.utils.data.importers.DatasetImporter) interface that defines how it imports datasets from disk when methods such as [Dataset.from\_importer()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.html#fiftyone.core.dataset.Dataset.from_importer) are used.

`DatasetImporter` itself is an abstract interface; the concrete interface that you should implement is determined by the type of dataset that you are importing. See [writing a custom DatasetImporter](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#custom-formats) for full details.

In this recipe, we’ll write a custom [LabeledImageDatasetImporter](https://voxel51.com/docs/fiftyone/api/fiftyone.utils.data.html#fiftyone.utils.data.importers.LabeledImageDatasetImporter) that can import an image classification dataset whose image metadata and labels are stored in a `labels.csv` file in the dataset directory with the following format:

```python
filepath,size_bytes,mime_type,width,height,num_channels,label
<filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
<filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
...

```

Here’s the complete definition of the `DatasetImporter`:

```python
[3]:

```

```python
import csv
import os

import fiftyone as fo
import fiftyone.utils.data as foud

class CSVImageClassificationDatasetImporter(foud.LabeledImageDatasetImporter):
    """Importer for image classification datasets whose filepaths and labels
    are stored on disk in a CSV file.

    Datasets of this type should contain a ``labels.csv`` file in their
    dataset directories in the following format::

        filepath,size_bytes,mime_type,width,height,num_channels,label
        <filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
        <filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
        ...

    Args:
        dataset_dir: the dataset directory
        shuffle (False): whether to randomly shuffle the order in which the
            samples are imported
        seed (None): a random seed to use when shuffling
        max_samples (None): a maximum number of samples to import. By default,
            all samples are imported
    """

    def __init__(
        self,
        dataset_dir,
        shuffle=False,
        seed=None,
        max_samples=None,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples
        )
        self._labels_file = None
        self._labels = None
        self._iter_labels = None

    def __iter__(self):
        self._iter_labels = iter(self._labels)
        return self

    def __next__(self):
        """Returns information about the next sample in the dataset.

        Returns:
            an  ``(image_path, image_metadata, label)`` tuple, where

            -   ``image_path``: the path to the image on disk
            -   ``image_metadata``: an
                :class:`fiftyone.core.metadata.ImageMetadata` instances for the
                image, or ``None`` if :meth:`has_image_metadata` is ``False``
            -   ``label``: an instance of :meth:`label_cls`, or a dictionary
                mapping field names to :class:`fiftyone.core.labels.Label`
                instances, or ``None`` if the sample is unlabeled

        Raises:
            StopIteration: if there are no more samples to import
        """
        (
            filepath,
            size_bytes,
            mime_type,
            width,
            height,
            num_channels,
            label,
        ) = next(self._iter_labels)

        image_metadata = fo.ImageMetadata(
            size_bytes=size_bytes,
            mime_type=mime_type,
            width=width,
            height=height,
            num_channels=num_channels,
        )

        label = fo.Classification(label=label)
        return filepath, image_metadata, label

    def __len__(self):
        """The total number of samples that will be imported.

        Raises:
            TypeError: if the total number is not known
        """
        return len(self._labels)

    @property
    def has_dataset_info(self):
        """Whether this importer produces a dataset info dictionary."""
        return False

    @property
    def has_image_metadata(self):
        """Whether this importer produces
        :class:`fiftyone.core.metadata.ImageMetadata` instances for each image.
        """
        return True

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) returned by this
        importer.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            importer is guaranteed to return labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the importer can produce a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the importer will return label dictionaries with keys
            and value-types specified by this dictionary. Not all keys need be
            present in the imported labels
        -   ``None``. In this case, the importer makes no guarantees about the
            labels that it may return
        """
        return fo.Classification

    def setup(self):
        """Performs any necessary setup before importing the first sample in
        the dataset.

        This method is called when the importer's context manager interface is
        entered, :func:`DatasetImporter.__enter__`.
        """
        labels_path = os.path.join(self.dataset_dir, "labels.csv")

        labels = []
        with open(labels_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append((
                    row["filepath"],
                    row["size_bytes"],
                    row["mime_type"],
                    row["width"],
                    row["height"],
                    row["num_channels"],
                    row["label"],
                ))

        # The `_preprocess_list()` function is provided by the base class
        # and handles shuffling/max sample limits
        self._labels = self._preprocess_list(labels)

    def close(self, *args):
        """Performs any necessary actions after the last sample has been
        imported.

        This method is called when the importer's context manager interface is
        exited, :func:`DatasetImporter.__exit__`.

        Args:
            *args: the arguments to :func:`DatasetImporter.__exit__`
        """
        pass

```

## Generating a sample dataset [¶](\#Generating-a-sample-dataset "Permalink to this headline")

In order to use `CSVImageClassificationDatasetImporter`, we need to generate a sample dataset in the required format.

Let’s first write a small utility to populate a `labels.csv` file in the required format.

```python
[4]:

```

```python
def write_csv_labels(samples, csv_path, label_field="ground_truth"):
    """Writes a labels CSV format for the given samples in the format expected
    by :class:`CSVImageClassificationDatasetImporter`.

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        csv_path: the path to write the CSV file
        label_field ("ground_truth"): the label field of the samples to write
    """
    # Ensure base directory exists
    basedir = os.path.dirname(csv_path)
    if basedir and not os.path.isdir(basedir):
        os.makedirs(basedir)

    # Write the labels
    with open(csv_path, "w") as f:
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
        for sample in samples:
            filepath = sample.filepath
            metadata = sample.metadata
            if metadata is None:
                metadata = fo.ImageMetadata.build_for(filepath)

            label = sample[label_field].label
            writer.writerow([\
                filepath,\
                metadata.size_bytes,\
                metadata.mime_type,\
                metadata.width,\
                metadata.height,\
                metadata.num_channels,\
                label,\
            ])

```

Now let’s populate a directory with a `labels.csv` file in the format required by `CSVImageClassificationDatasetImporter` with some samples from the test split of CIFAR-10:

```python
[7]:

```

```python
import fiftyone.zoo as foz

dataset_dir = "/tmp/fiftyone/custom-dataset-importer"
num_samples = 1000

#
# Load `num_samples` from CIFAR-10
#
# This command will download the test split of CIFAR-10 from the web the first
# time it is executed, if necessary
#
cifar10_test = foz.load_zoo_dataset("cifar10", split="test")
samples = cifar10_test.limit(num_samples)

# This dataset format requires samples to have their `metadata` fields populated
print("Computing metadata for samples")
samples.compute_metadata()

# Write labels to disk in CSV format
csv_path = os.path.join(dataset_dir, "labels.csv")
print("Writing labels for %d samples to '%s'" % (num_samples, csv_path))
write_csv_labels(samples, csv_path)

```

```python
Split 'test' already downloaded
Loading existing dataset 'cifar10-test'. To reload from disk, first delete the existing dataset
Computing metadata for samples
 100% |█████| 1000/1000 [421.2ms elapsed, 0s remaining, 2.4K samples/s]
Writing labels for 1000 samples to '/tmp/fiftyone/custom-dataset-importer/labels.csv'

```

Let’s inspect the contents of the labels CSV to ensure they’re in the correct format:

```python
[13]:

```

```python
!head -n 10 /tmp/fiftyone/custom-dataset-importer/labels.csv

```

```python
filepath,size_bytes,mime_type,width,height,num_channels,label
~/fiftyone/cifar10/test/data/000001.jpg,1422,image/jpeg,32,32,3,cat
~/fiftyone/cifar10/test/data/000002.jpg,1285,image/jpeg,32,32,3,ship
~/fiftyone/cifar10/test/data/000003.jpg,1258,image/jpeg,32,32,3,ship
~/fiftyone/cifar10/test/data/000004.jpg,1244,image/jpeg,32,32,3,airplane
~/fiftyone/cifar10/test/data/000005.jpg,1388,image/jpeg,32,32,3,frog
~/fiftyone/cifar10/test/data/000006.jpg,1311,image/jpeg,32,32,3,frog
~/fiftyone/cifar10/test/data/000007.jpg,1412,image/jpeg,32,32,3,automobile
~/fiftyone/cifar10/test/data/000008.jpg,1218,image/jpeg,32,32,3,frog
~/fiftyone/cifar10/test/data/000009.jpg,1262,image/jpeg,32,32,3,cat

```

## Importing a dataset [¶](\#Importing-a-dataset "Permalink to this headline")

With our dataset and `DatasetImporter` in-hand, loading the data as a FiftyOne dataset is as simple as follows:

```python
[14]:

```

```python
# Import the dataset
print("Importing dataset from '%s'" % dataset_dir)
importer = CSVImageClassificationDatasetImporter(dataset_dir)
dataset = fo.Dataset.from_importer(importer)

```

```python
Importing dataset from '/tmp/fiftyone/custom-dataset-importer'
 100% |█████| 1000/1000 [780.7ms elapsed, 0s remaining, 1.3K samples/s]

```

```python
[15]:

```

```python
# Print summary information about the dataset
print(dataset)

```

```python
Name:           2020.07.14.22.33.01
Persistent:     False
Num samples:    1000
Tags:           []
Sample fields:
    filepath:     fiftyone.core.fields.StringField
    tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)

```

```python
[16]:

```

```python
# Print a sample
print(dataset.first())

```

```python
<Sample: {
    'dataset_name': '2020.07.14.22.33.01',
    'id': '5f0e6add1dfd5f8c299ac528',
    'filepath': '~/fiftyone/cifar10/test/data/000001.jpg',
    'tags': BaseList([]),
    'metadata': <ImageMetadata: {
        'size_bytes': 1422,
        'mime_type': 'image/jpeg',
        'width': 32,
        'height': 32,
        'num_channels': 3,
    }>,
    'ground_truth': <Classification: {'label': 'cat', 'confidence': None, 'logits': None}>,
}>

```

## Cleanup [¶](\#Cleanup "Permalink to this headline")

You can cleanup the files generated by this recipe by running:

```python
[17]:

```

```python
!rm -rf /tmp/fiftyone

```

