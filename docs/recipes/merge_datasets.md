# Merging Datasets [¶](\#Merging-Datasets "Permalink to this headline")

This recipe demonstrates a simple pattern for merging FiftyOne Datasets via [Dataset.merge\_samples()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html?highlight=merge_samples#fiftyone.core.dataset.Dataset.merge_samples).

Merging datasets is an easy way to:

- Combine multiple datasets with information about the same underlying raw media (images and videos)

- Add model predictions to a FiftyOne dataset, to compare with ground truth annotations and/or other models


## Setup [¶](\#Setup "Permalink to this headline")

If you haven’t already, install FiftyOne:

```python
[ ]:

```

```python
!pip install fiftyone

```

In this recipe, we’ll work with a dataset downloaded from the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/zoo.html).

To access the dataset, install `torch` and `torchvision`, if necessary:

```python
[ ]:

```

```python
!pip install torch torchvision

```

Then download the test split of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html):

```python
[1]:

```

```python
# Download the validation split of COCO-2017
!fiftyone zoo datasets download cifar10 --splits test

```

```python
Split 'test' already downloaded

```

## Merging model predictions [¶](\#Merging-model-predictions "Permalink to this headline")

Load the test split of CIFAR-10 into FiftyOne:

```python
[1]:

```

```python
import random
import os

import fiftyone as fo
import fiftyone.zoo as foz

# Load test split of CIFAR-10
dataset = foz.load_zoo_dataset("cifar10", split="test", dataset_name="merge-example")
classes = dataset.info["classes"]

print(dataset)

```

```python
Split 'test' already downloaded
Loading 'cifar10' split 'test'
 100% |███| 10000/10000 [14.1s elapsed, 0s remaining, 718.2 samples/s]
Name:           merge-example
Media type:     image
Num samples:    10000
Persistent:     False
Info:           {'classes': ['airplane', 'automobile', 'bird', ...]}
Tags:           ['test']
Sample fields:
    filepath:     fiftyone.core.fields.StringField
    tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)

```

The dataset contains ground truth labels in its `ground_truth` field:

```python
[2]:

```

```python
# Print a sample from the dataset
print(dataset.first())

```

```python
<Sample: {
    'id': '5fee1a40f653ce52a9d077b1',
    'media_type': 'image',
    'filepath': '/Users/Brian/fiftyone/cifar10/test/data/000001.jpg',
    'tags': BaseList(['test']),
    'metadata': None,
    'ground_truth': <Classification: {
        'id': '5fee1a40f653ce52a9d077b0',
        'label': 'horse',
        'confidence': None,
        'logits': None,
    }>,
}>

```

Suppose you would like to add model predictions to some samples from the dataset.

The usual way to do this is to just iterate over the dataset and add your predictions directly to the samples:

```python
[3]:

```

```python
def run_inference(filepath):
    # Run inference on `filepath` here.
    # For simplicity, we'll just generate a random label
    label = random.choice(classes)

    return fo.Classification(label=label)

```

```python
[4]:

```

```python
# Choose 100 samples at random
random_samples = dataset.take(100)

# Add model predictions to dataset
for sample in random_samples:
    sample["predictions"] = run_inference(sample.filepath)
    sample.save()

print(dataset)

```

```python
Name:           merge-example
Media type:     image
Num samples:    10000
Persistent:     False
Info:           {'classes': ['airplane', 'automobile', 'bird', ...]}
Tags:           ['test']
Sample fields:
    filepath:     fiftyone.core.fields.StringField
    tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
    predictions:  fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)

```

However, suppose you store the predictions in a separate dataset:

```python
[5]:

```

```python
# Filepaths of images to proces
filepaths = [s.filepath for s in dataset.take(100)]

# Run inference
predictions = fo.Dataset()
for filepath in filepaths:
    sample = fo.Sample(filepath=filepath)

    sample["predictions"] = run_inference(filepath)

    predictions.add_sample(sample)

print(predictions)

```

```python
Name:           2020.12.31.12.37.09
Media type:     image
Num samples:    100
Persistent:     False
Info:           {}
Tags:           []
Sample fields:
    filepath:    fiftyone.core.fields.StringField
    tags:        fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:    fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    predictions: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)

```

You can easily merge the `predictions` dataset into the main dataset via [Dataset.merge\_samples()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html?highlight=merge_samples#fiftyone.core.dataset.Dataset.merge_samples).

Let’s start by creating a fresh copy of CIFAR-10 that doesn’t have predictions:

```python
[6]:

```

```python
dataset2 = dataset.exclude_fields("predictions").clone(name="merge-example2")
print(dataset2)

```

```python
Name:           merge-example2
Media type:     image
Num samples:    10000
Persistent:     False
Info:           {'classes': ['airplane', 'automobile', 'bird', ...]}
Tags:           ['test']
Sample fields:
    filepath:     fiftyone.core.fields.StringField
    tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)

```

Now let’s merge the predictions into the fresh dataset:

```python
[7]:

```

```python
# Merge predictions
dataset2.merge_samples(predictions)

# Verify that 100 samples in `dataset2` now have predictions
print(dataset2.exists("predictions"))

```

```python
Dataset:        merge-example2
Media type:     image
Num samples:    100
Tags:           []
Sample fields:
    filepath:     fiftyone.core.fields.StringField
    tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
    predictions:  fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
View stages:
    1. Exists(field='predictions', bool=True)

```

Let’s print a sample with predictions to verify that the merge happened as expected:

```python
[8]:

```

```python
# Print a sample with predictions
print(dataset2.exists("predictions").first())

```

```python
<SampleView: {
    'id': '5fee1a40f653ce52a9d07883',
    'media_type': 'image',
    'filepath': '/Users/Brian/fiftyone/cifar10/test/data/000071.jpg',
    'tags': BaseList([]),
    'metadata': None,
    'ground_truth': <Classification: {
        'id': '5fee1a40f653ce52a9d07882',
        'label': 'frog',
        'confidence': None,
        'logits': None,
    }>,
    'predictions': <Classification: {
        'id': '5fee1a56f653ce52a9d0ee71',
        'label': 'horse',
        'confidence': None,
        'logits': None,
    }>,
}>

```

## Customizing the merge key [¶](\#Customizing-the-merge-key "Permalink to this headline")

By default, samples with the same absolute `filepath` are merged. However, you can customize this as desired via various keyword arguments of [Dataset.merge\_samples()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html?highlight=merge_samples#fiftyone.core.dataset.Dataset.merge_samples).

For example, the command below will merge samples with the same base filename, ignoring the directory:

```python
[9]:

```

```python
# Create another fresh dataset to work with
dataset3 = dataset.exclude_fields("predictions").clone(name="merge-example3")

# Merge predictions, using the base filename of the samples to decide which samples to merge
# In this case, we've already performed the merge, so the existing data is overwritten
key_fcn = lambda sample: os.path.basename(sample.filepath)

dataset3.merge_samples(predictions, key_fcn=key_fcn)

```

```python
Indexing dataset...
 100% |███| 10000/10000 [3.6s elapsed, 0s remaining, 2.8K samples/s]
Merging samples...
 100% |███████| 100/100 [348.5ms elapsed, 0s remaining, 287.0 samples/s]

```

Let’s print a sample with predictions to verify that the merge happened as expected:

```python
[10]:

```

```python
# Print a sample with predictions
print(dataset3.exists("predictions").first())

```

```python
<SampleView: {
    'id': '5fee1a40f653ce52a9d07883',
    'media_type': 'image',
    'filepath': '/Users/Brian/fiftyone/cifar10/test/data/000071.jpg',
    'tags': BaseList([]),
    'metadata': None,
    'ground_truth': <Classification: {
        'id': '5fee1a40f653ce52a9d07882',
        'label': 'frog',
        'confidence': None,
        'logits': None,
    }>,
    'predictions': <Classification: {
        'id': '5fee1a56f653ce52a9d0ee71',
        'label': 'horse',
        'confidence': None,
        'logits': None,
    }>,
}>

```

