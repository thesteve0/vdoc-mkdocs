# Evaluating a Classifier with FiftyOne [¶](\#Evaluating-a-Classifier-with-FiftyOne "Permalink to this headline")

This notebook demonstrates an end-to-end example of fine-tuning a classification model [using fastai](https://github.com/fastai/fastai) on a [Kaggle dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) and using FiftyOne to evaluate it and understand the strengths and weaknesses of both the model and the underlying ground truth annotations.

Specifically, we’ll cover:

- Downloading the dataset via the [Kaggle API](https://github.com/Kaggle/kaggle-api)

- Loading the dataset [into FiftyOne](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/index.html)

- Indexing the dataset by uniqueness using FiftyOne’s [uniqueness method](https://voxel51.com/docs/fiftyone/user_guide/brain.html#image-uniqueness) to identify interesting visual characteristics

- Fine-tuning a model on the dataset [using fastai](https://github.com/fastai/fastai)

- [Evaluating](https://voxel51.com/docs/fiftyone/user_guide/evaluation.html) the fine-tuned model using FiftyOne

- [Exporting](https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html) the FiftyOne dataset for offline analysis


**So, what’s the takeaway?**

The loss function of your model training loop alone doesn’t give you the full picture of a model. In practice, the limiting factor on your model’s performance is often data quality issues that FiftyOne can help you address. In this notebook, we’ll cover:

- Viewing the _most unique_ incorrect samples using FiftyOne’s [uniqueness method](https://voxel51.com/docs/fiftyone/user_guide/brain.html#image-uniqueness)

- Viewing the _hardest_ incorrect predictions using FiftyOne’s [hardness method](https://voxel51.com/docs/fiftyone/user_guide/brain.html#sample-hardness)

- Identifying ground truth _mistakes_ using FiftyOne’s [mistakenness method](https://voxel51.com/docs/fiftyone/user_guide/brain.html#label-mistakes)


Running the workflow presented here on your ML projects will help you to understand the current failure modes (edge cases) of your model and how to fix them, including:

- Identifying scenarios that require additional training samples in order to boost your model’s performance

- Deciding whether your ground truth annotations have errors/weaknesses that need to be corrected before any subsequent model training will be profitable


## Setup [¶](\#Setup "Permalink to this headline")

If you haven’t already, install FiftyOne:

```python
[ ]:

```

```python
!pip install fiftyone

```

We’ll also need `torch` and `torchvision` installed:

```python
[1]:

```

```python
!pip install torch torchvision

```

## Download dataset [¶](\#Download-dataset "Permalink to this headline")

Let’s start by downloading the [Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) from Kaggle using the [Kaggle API](https://github.com/Kaggle/kaggle-api):

```python
[ ]:

```

```python
!pip install --upgrade kaggle

```

```python
[4]:

```

```python
%%bash

# You can create an account for free and get an API token as follows:
# kaggle.com > account > API > Create new API token
export KAGGLE_USERNAME=XXXXXXXXXXXXXXXX
export KAGGLE_KEY=XXXXXXXXXXXXXXXX

kaggle datasets download -d iarunava/cell-images-for-detecting-malaria

```

```python
Downloading cell-images-for-detecting-malaria.zip

```

```python
100%|██████████| 675M/675M [00:23<00:00, 30.7MB/s]

```

```python
[5]:

```

```python
%%bash

unzip -q cell-images-for-detecting-malaria.zip

rm -rf cell_images/cell_images
rm cell_images/Parasitized/Thumbs.db
rm cell_images/Uninfected/Thumbs.db
rm cell-images-for-detecting-malaria.zip

```

The unzipped dataset consists of a `cell_images/` folder with two subdirectories— `Uninfected` and `Parasitized`—that each contain 13782 example images of the respective class of this binary classification task:

```python
[6]:

```

```python
%%bash

ls -lah cell_images/Uninfected | head
ls -lah cell_images/Parasitized | head

printf "\nClass counts\n"
ls -lah cell_images/Uninfected | wc -l
ls -lah cell_images/Parasitized | wc -l

```

```python
total 354848
drwxr-xr-x  13781 voxel51  staff   431K Feb 18 08:56 .
drwxr-xr-x      4 voxel51  staff   128B Feb 18 08:56 ..
-rw-r--r--      1 voxel51  staff    11K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_128.png
-rw-r--r--      1 voxel51  staff    11K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_131.png
-rw-r--r--      1 voxel51  staff   9.7K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_144.png
-rw-r--r--      1 voxel51  staff   5.8K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_21.png
-rw-r--r--      1 voxel51  staff   9.4K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_25.png
-rw-r--r--      1 voxel51  staff   7.5K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_34.png
-rw-r--r--      1 voxel51  staff    10K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_48.png
total 404008
drwxr-xr-x  13781 voxel51  staff   431K Feb 18 08:56 .
drwxr-xr-x      4 voxel51  staff   128B Feb 18 08:56 ..
-rw-r--r--      1 voxel51  staff    14K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_162.png
-rw-r--r--      1 voxel51  staff    18K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_163.png
-rw-r--r--      1 voxel51  staff    13K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_164.png
-rw-r--r--      1 voxel51  staff    13K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_165.png
-rw-r--r--      1 voxel51  staff    11K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_166.png
-rw-r--r--      1 voxel51  staff    14K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_167.png
-rw-r--r--      1 voxel51  staff    11K Oct 14  2019 C100P61ThinF_IMG_20150918_144104_cell_168.png

Class counts
   13782
   13782

```

## Load dataset into FiftyOne [¶](\#Load-dataset-into-FiftyOne "Permalink to this headline")

Let’s load the dataset into [FiftyOne](https://voxel51.com/docs/fiftyone) and explore it!

```python
[ ]:

```

```python
import os
import fiftyone as fo

DATASET_DIR = os.path.join(os.getcwd(),"cell_images/")

```

### Create FiftyOne dataset [¶](\#Create-FiftyOne-dataset "Permalink to this headline")

FiftyOne provides builtin support for loading datasets in [dozens of common formats](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/index.html) with a single line of code:

```python
[ ]:

```

```python
# Create FiftyOne dataset
dataset = fo.Dataset.from_dir(
    DATASET_DIR,
    fo.types.ImageClassificationDirectoryTree,
    name="malaria-cell-images",
)
dataset.persistent = True

print(dataset)

```

```python
 100% |███| 27558/27558 [35.8s elapsed, 0s remaining, 765.8 samples/s]
Name:           malaria-cell-images
Media type:     image
Num samples:    27558
Persistent:     True
Info:           {'classes': ['Parasitized', 'Uninfected']}
Tags:           []
Sample fields:
    filepath:     fiftyone.core.fields.StringField
    tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)

```

### (Future use) Load an existing FiftyOne dataset [¶](\#(Future-use)-Load-an-existing-FiftyOne-dataset "Permalink to this headline")

Now that the data is loaded into FiftyOne, you can easily [work with](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html) the same dataset in a future session on the same machine by loading it by name:

```python
[ ]:

```

```python
# Load existing dataset
dataset = fo.load_dataset("malaria-cell-images")
print(dataset)

```

### Index the dataset by visual uniqueness [¶](\#Index-the-dataset-by-visual-uniqueness "Permalink to this headline")

Let’s start by indexing the dataset by visual uniqueness using FiftyOne’s [image uniqueness method](https://voxel51.com/docs/fiftyone/user_guide/brain.html#image-uniqueness).

This method adds a scalar `uniqueness` field to each sample that measures the relative visual uniqueness of each sample compared to the other samples in the dataset.

```python
[ ]:

```

```python
import fiftyone.brain as fob

fob.compute_uniqueness(dataset)

```

```python
Loading uniqueness model...
Downloading model from Google Drive ID '1SIO9XreK0w1ja4EuhBWcR10CnWxCOsom'...
 100% |████|  100.6Mb/100.6Mb [135.7ms elapsed, 0s remaining, 741.3Mb/s]
Preparing data...
Generating embeddings...
 100% |███| 27558/27558 [39.6s elapsed, 0s remaining, 618.6 samples/s]
Computing uniqueness...
Saving results...
 100% |███| 27558/27558 [42.9s elapsed, 0s remaining, 681.0 samples/s]
Uniqueness computation complete

```

### Visualize dataset in the App [¶](\#Visualize-dataset-in-the-App "Permalink to this headline")

Now let’s launch the [FiftyOne App](https://voxel51.com/docs/fiftyone/user_guide/app.html) and use it to interactively explore the dataset.

For example, try using the [view bar](https://voxel51.com/docs/fiftyone/user_guide/app.html#using-the-view-bar) to sort the samples so that we can view the _most visually unique_ samples in the dataset:

```python
[2]:

```

```python
# Most of the MOST UNIQUE samples are parasitized
session = fo.launch_app(dataset)

```

Activate

![](<Base64-Image-Removed>)

Now let’s add a `Limit(500)` stage in the view bar and open the `Labels` tab to view some statistics about the 500 most unique samples in the dataset.

Notice that a vast majority of the most visually unique samples in the dataset are `Parasitized`, which makes sense because these are the infected, abnormal cells.

```python
[6]:

```

```python
session.show()

```

Activate

![](<Base64-Image-Removed>)

Conversely, if we use the view bar to show the 500 _least visually unique_ samples, we find that 499 of them are `Uninfected`!

```python
[7]:

```

```python
# All of the LEAST UNIQUE samples are uninfected
session.show()

```

Activate

![](<Base64-Image-Removed>)