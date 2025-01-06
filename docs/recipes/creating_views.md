Table of Contents

- [Docs](../index.html) >

- [FiftyOne Recipes](index.html) >
- Creating Views

Contents


# Creating Views [¶](\#Creating-Views "Permalink to this headline")

[FiftyOne datasets](https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html) provide the flexibility to store large, complex data. While it is helpful that data can be imported and exported easily, the real potential of FiftyOne comes from its powerful query language that you can use to define custom **views** into your datasets.

A [dataset view](https://voxel51.com/docs/fiftyone/user_guide/using_views.html) can be thought of as a pipeline of operations that is applied to a dataset to extract a subset of the dataset whose samples and fields are filtered, sorted, shuffled, etc. Check out [this page](https://voxel51.com/docs/fiftyone/user_guide/using_views.html) for an extended discussion of dataset views.

In this notebook, we’ll do a brief walkthrough of creating and using dataset views.

## Setup [¶](\#Setup "Permalink to this headline")

If you haven’t already, install FiftyOne:

```
[ ]:

```

```
!pip install fiftyone

```

## Overview [¶](\#Overview "Permalink to this headline")

To start out, lets import FiftyOne, load up a dataset, and evaluate some predicted object detections.

```
[ ]:

```

```
import fiftyone as fo
import fiftyone.zoo as foz

```

```
[ ]:

```

```
dataset = foz.load_zoo_dataset("quickstart")
dataset.evaluate_detections("predictions", gt_field="ground_truth", eval_key="eval")

```

Dataset views can range from as simple as “select a slice of the dataset” to “filter sample that have at least two large bounding boxes of people or dogs with high confidence and that were evaluated to be a false positive, then crop all images to those bounding boxes”:

```
[ ]:

```

```
from fiftyone import ViewField as F

# Slice dataset
simple_view = dataset[51:151]

# Complex filtering and conversion
complex_view = (
    dataset
    .filter_labels(
        "predictions", (
            (F("confidence") > 0.7)
            & ((F("bounding_box")[2] * F("bounding_box")[3]) > 0.3)
            & (F("eval") == "fp")
            & (F("label").is_in(["person", "dog"]))
        )
    ).match(
        F("predictions.detections").length() > 2
    ).to_patches("predictions")
)

```

The goal is that, by the end of this notebook, creating complex views like the one above will be as straight forward as the simple views.

## View basics [¶](\#View-basics "Permalink to this headline")

“Creating a view from a dataset” is simply the process of performing an operation on a dataset that returns a `DatasetView`. The most basic way to turn a `Dataset` into a `DatasetView` is to just call `view()`.

```
[ ]:

```

```
# A view that contains the entire dataset
view = dataset.view()

```

Within FiftyOne, views and datasets are largely interchangable in nearly all operations. Anything you can do to a dataset, you can also do to a view.

```
[ ]:

```

```
print(view)

```

```
Dataset:     quickstart
Media type:  image
Num samples: 200
Tags:        ['validation']
Sample fields:
    id:              fiftyone.core.fields.ObjectIdField
    filepath:        fiftyone.core.fields.StringField
    tags:            fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:        fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth:    fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
    uniqueness:      fiftyone.core.fields.FloatField
    predictions:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
    eval_tp:         fiftyone.core.fields.IntField
    eval_fp:         fiftyone.core.fields.IntField
    eval_fn:         fiftyone.core.fields.IntField
    is_cloudy:       fiftyone.core.fields.BooleanField
    classification:  fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
    classifications: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
View stages:
    ---

```

To create some more interesting views, you need to apply a view stage operation to the dataset. The list of available view stages can be printed as follows:

```
[ ]:

```

```
dataset.list_view_stages()

```

```
['exclude',\
 'exclude_by',\
 'exclude_fields',\
 'exclude_frames',\
 'exclude_labels',\
 'exists',\
 'filter_field',\
 'filter_labels',\
 'filter_classifications',\
 'filter_detections',\
 'filter_polylines',\
 'filter_keypoints',\
 'geo_near',\
 'geo_within',\
 'group_by',\
 'limit',\
 'limit_labels',\
 'map_labels',\
 'set_field',\
 'match',\
 'match_frames',\
 'match_labels',\
 'match_tags',\
 'mongo',\
 'select',\
 'select_by',\
 'select_fields',\
 'select_frames',\
 'select_labels',\
 'shuffle',\
 'skip',\
 'sort_by',\
 'sort_by_similarity',\
 'take',\
 'to_patches',\
 'to_evaluation_patches',\
 'to_clips',\
 'to_frames']

```

These view stages allow you to perform many useful operations on datasets like slicing, sorting, shuffling, filtering, and more.

For example, the [take()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.take) stage lets you extract a random subset of samples from the dataset:

```
[ ]:

```

```
random_view = dataset.take(100)

print(len(random_view))

```

```
100

```

These view stages can also be chained together, each operating on the view returned by the previous stage:

```
[ ]:

```

```
sorted_random_view = random_view.sort_by("filepath")
sliced_sorted_random_view = sorted_random_view[10:51]

```

Note that the slicing syntax is simply a different representation of the [skip()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.skip) and [limit()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.limit) stages:

```
[ ]:

```

```
sliced_sorted_random_view = sorted_random_view.skip(10).limit(41)

```

An example of one of the stages used in this notebook is [match()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.match). This stage will keep or remove samples in the dataset one by one based on if some expression applied to the sample resolves to True or False.

For example, we can create a view that includes all samples with a uniqueness greater than 0.5:

```
[ ]:

```

```
matched_view = dataset.match(F("uniqueness") > 0.5)

```

Another useful view stage is [set\_field()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.set_field). This stage will actually modify a field in your dataset based on the provided expression. Note that this modification is only within the resulting `DatasetView` and will not modify the underlying dataset.

For example, lets set a boolean field called `is_cloudy` to True for all samples in the dataset. Note that when using `set_field()`, you need to ensure that the field exists on the dataset first.

```
[ ]:

```

```
dataset.add_sample_field("is_cloudy", fo.BooleanField)
cloudy_view = dataset.set_field("is_cloudy", True)

dataset.set_values("is_cloudy", [True]*len(dataset))

```

## View expressions [¶](\#View-expressions "Permalink to this headline")

At this point, you might be wondering “what is this `F` that I keep seeing everywhere”? That `F` defines a [ViewField](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewField) which can be used to write a [ViewExpression](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewExpression). These expressions are what give you the power to write custom queries based on information that exists in
your dataset.

In this section, we go over what some view expression operations and how to write more complex views.

Most view stages accept a [ViewExpression](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewExpression) as input. View stages that seemingly operate on fields can also accept expressions! For example, [sort\_by()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.sort_by) can accept a field name or an expression:

```
[ ]:

```

```
# Sort by filepaths
dataset.sort_by("filepath")

# Sort by the number of predicted objects per sample
dataset.sort_by(F("predictions.detections").length())

```

```
Dataset:     quickstart
Media type:  image
Num samples: 200
Tags:        ['validation']
Sample fields:
    id:           fiftyone.core.fields.ObjectIdField
    filepath:     fiftyone.core.fields.StringField
    tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
    uniqueness:   fiftyone.core.fields.FloatField
    predictions:  fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
    eval_tp:      fiftyone.core.fields.IntField
    eval_fp:      fiftyone.core.fields.IntField
    eval_fn:      fiftyone.core.fields.IntField
    is_cloudy:    fiftyone.core.fields.BooleanField
View stages:
    1. SortBy(field_or_expr={'$size': {'$ifNull': [...]}}, reverse=False)

```

The idea is to think about what is expected by a view stage, then provide the input that is needed in the form of a string or an expression.

[sort\_by()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.sort_by) operates on a sample-level, meaning we can either provide it the name of a sample-level field to use for sorting ( `filepath`) or we can provide it an expression that _resolves to_ a sample-level value. In this case the expression is counting the number of predicted objects for each sample and using those integers to sort the dataset.

### View fields [¶](\#View-fields "Permalink to this headline")

As mentioned, view expressions are built around view fields. A [ViewField](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewField) is how you inject the information stored in a specific field of your dataset into a view expression.

For example, if you had a boolean field on your dataset called `is_cloudy` indicating if the image contains cloudy or not, then for each sample, `F("is_cloudy")` can be thought of as being replaced with the value of `"is_cloudy"` for that sample. Since values in the field are themselves boolean, the view to match samples where `"is_cloudy"` is True is simply:

```
cloudy_view = dataset.match(F("is_cloudy"))

```

In our dataset, after performing evaluation, we populated the field `eval_tp` on each sample with is an integer containing the number of true positive predictions exist in the sample. There are multiple ways to match samples based on the `eval_tp` field.

The way to think about view expressions in this case is the same as the expressions for the if-statement in Python that resolve in a boolean context.

```
[ ]:

```

```
a = True
b = 51

if a: # Nothing else needed
    pass

if b > 4:
    # True if b > 4
    pass

if b:
    # True if b != 0
    pass

```

```
[ ]:

```

```
tp_view = dataset.match(F("eval_tp") > 4)

print(len(tp_view))

```

```
69

```

When providing just an integer in the expression in a Python if-statement, then the statement is True as long as the integer is not zero. The same logic applies with view expressions in this case:

```
[ ]:

```

```
nonzero_tp_view = dataset.match(F("eval_tp"))

print(len(nonzero_tp_view))

```

```
198

```

We can also use `~` to negate an expression:

```
[ ]:

```

```
zero_tp_view = dataset.match(~F("eval_tp"))

print(zero_tp_view.values("eval_tp"))

```

```
[0, 0]

```

### Nested lists [¶](\#Nested-lists "Permalink to this headline")

The most difficult/subtle aspect of creating view expressions is how to handle nested lists.

To get a better idea of which samples contain lists, you can print out your sample as a dictionary:

```
[ ]:

```

```
sample = fo.Sample(
    filepath="example.png",
    ground_truth=fo.Detections(
        detections=[\
            fo.Detection(label="cat", bounding_box=[0.1, 0.1, 0.8, 0.8])\
        ]
    ),
)

fo.pprint(sample.to_dict())

```

```
{
    'filepath': '/content/example.png',
    'tags': [],
    'metadata': None,
    'ground_truth': {
        '_cls': 'Detections',
        'detections': [\
            {\
                '_id': {'$oid': '622f67345627ae9fa020e6f9'},\
                '_cls': 'Detection',\
                'attributes': {},\
                'tags': [],\
                'label': 'cat',\
                'bounding_box': [0.1, 0.1, 0.8, 0.8],\
            },\
        ],
    },
}

```

Here you can see that `ground_truth.detections` is a list.

If you have a field containing a primitive value, then it rarely requires more than one operation to get the value that is needed by the view stage. However, when working with a list of values in a field, then there can be multiple different operations that need to be performed to get to the desired value.

The most important operations for working with lists are:

- [filter()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewExpression.filter): apply a boolean to each element of a list to determine what to keep, resolving to a list

- [map()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewExpression.map): apply a function to each element of a list, resolving to a list

- [reduce()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewExpression.reduce): operates on a list and resolves to a single value


### Filtering list fields [¶](\#Filtering-list-fields "Permalink to this headline")

The [filter()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewExpression.filter) operation is quite useful to allow for fine-grained access to the information that is to be kept and removed from the view.

```
[ ]:

```

```
# Only include predictions with `confidence` of at least 0.9
view = dataset.set_field(
    "predictions.detections",
    F("detections").filter(F("confidence") > 0.9)
)

```

Note that the [filter\_labels()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.filter_labels) operation is simply a simplification of the filter operation and [set\_field()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.set_field). This operation will automatically apply the given expression to the corresponding list field of the label if applicable ( `Detections`, `Classifications`, etc) or
will apply the expression as a match operation for non-list labels ( `Detection`, `Classification`, etc).

```
[ ]:

```

```
# Filter detections
view1 = dataset.filter_labels("ground_truth", F("label") == "cat")

# Equivalently
view2 = (
    dataset
    .set_field("ground_truth.detections", F("detections").filter(F("label") == "cat"))
    .match(F("ground_truth.detections").length() > 0)
)

print(len(view1))
print(len(view2))

```

```
14
14

```

The match operation above was added since by default, [filter\_labels()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.filter_labels) sets the keyword argument `only_matches=True`.

```
[ ]:

```

```
# Add example classification labels
dataset.set_values("classifications", [fo.Classification(label="cat")]*len(dataset))

# Filter classification
view1 = dataset.filter_labels("classifications", F("label") == "cat")

# Equivalently
view2 = dataset.match(F("classifications.label") == "cat")

print(len(view1))
print(len(view2))

```

```
200
200

```

### Mapping list fields [¶](\#Mapping-list-fields "Permalink to this headline")

The [map()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewExpression.map) operation can be used to apply an expression to every element of a list. For example, we can update the tags to set every tag to uppercase:

```
[ ]:

```

```
transform_tag = F().upper()
view = dataset.set_field("tags", F("tags").map(transform_tag))

print(view)

```

```
Dataset:     quickstart
Media type:  image
Num samples: 200
Tags:        ['VALIDATION']
Sample fields:
    id:              fiftyone.core.fields.ObjectIdField
    filepath:        fiftyone.core.fields.StringField
    tags:            fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:        fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.Metadata)
    ground_truth:    fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
    uniqueness:      fiftyone.core.fields.FloatField
    predictions:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)
    eval_tp:         fiftyone.core.fields.IntField
    eval_fp:         fiftyone.core.fields.IntField
    eval_fn:         fiftyone.core.fields.IntField
    is_cloudy:       fiftyone.core.fields.BooleanField
    classification:  fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
    classifications: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Classification)
View stages:
    1. SetField(field='tags', expr={'$map': {'as': 'this', 'in': {...}, 'input': '$tags'}})

```

Note that the `F()` above is empty, indicating that [upper()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewExpression.upper) is applied to the primitives stored in each element of the field. In this case, the primitives are the string tags. In general, `F()` references the root of the current context.

### Reducing list fields [¶](\#Reducing-list-fields "Permalink to this headline")

The [reduce()](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewField.reduce) operation lets you take a list, operate on each element of it, and return some value. Reduce expressions generally involve some `VALUE` that is being aggregated as each element is iterated over. For example, this could be some float that values are added to, a string that gets concatenated each iteration, or even a list to which elements are appended.

Say that we want to set a field on our predictions containing the IDs of the corresponding ground truth objects that were matched to the true positives. We can use filter and reduce to accomplish this as follows:

```
[ ]:

```

```
from fiftyone.core.expressions import VALUE

```

```
[ ]:

```

```
# Get all of the matched gt object ids
view = (
    dataset
    .set_field(
        "predictions.gt_ids",
        F("detections")
        .filter(F("eval") == "tp")
        .reduce(VALUE.append(F("eval_id")), init_val=[])
    )
)
view.first().predictions.gt_ids

```

```
['5f452471ef00e6374aac53c8', '5f452471ef00e6374aac53ca']

```

### Referencing root fields [¶](\#Referencing-root-fields "Permalink to this headline")

Another useful property of expressions is prepending your field names with `$` to refer to the root of the document. This can be used, for example, to use sample-level information like `metadata` when filtering at a detection-level:

```
[ ]:

```

```
dataset.compute_metadata()

# Computes the area of each bounding box in pixels
bbox_area = (
    F("$metadata.width") * F("bounding_box")[2] *
    F("$metadata.height") * F("bounding_box")[3]
)

# Only contains boxes whose area is between 32^2 and 96^2 pixels
medium_boxes_view = dataset.filter_labels(
    "predictions", (32 ** 2 < bbox_area) & (bbox_area < 96 ** 2)
)

```

For a complete listing of all operations that can be performed to create view expressions and examples of each, [check out the API documentation](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html).

## Aggregations [¶](\#Aggregations "Permalink to this headline")

[Aggregations](https://voxel51.com/docs/fiftyone/user_guide/using_aggregations.html) provide a convenient syntax to compute aggregate statistics or extract values across a dataset or view.

For example, you can use aggregations to get information like:

- The boundary values of a field

- The unique label names in your dataset

- The standard deviation of a value across your samples

- Extract a slice of field values across a view


You can view the available aggregations like so:

```
[ ]:

```

```
dataset.list_aggregations()

```

```
['bounds',\
 'count',\
 'count_values',\
 'distinct',\
 'histogram_values',\
 'mean',\
 'std',\
 'sum',\
 'values']

```

[The documentation](https://voxel51.com/docs/fiftyone/user_guide/using_aggregations.html) already contains plenty of detailed information about aggregations. This section just highlights how view expressions can be used with aggregations.

In the simplest case, aggregations can be performed by providing the name of a field you want to compute on:

```
[ ]:

```

```
print(dataset.distinct("predictions.detections.label"))

```

```
['airplane', 'apple', 'backpack', 'banana', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

```

However, you can also pass a [ViewExpression](https://voxel51.com/docs/fiftyone/api/fiftyone.core.expressions.html#fiftyone.core.expressions.ViewExpression) to the aggregation method, in which case the expression will be evaluated and then aggregated as requested:

```
[ ]:

```

```
print(dataset.distinct(F("uniqueness").round(2)))

```

```
[0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.34, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.72, 0.73, 0.74, 0.75, 0.78, 0.8, 0.82, 0.92, 1.0]

```

## Summary [¶](\#Summary "Permalink to this headline")

Dataset views and the view expressions language are powerful and flexible aspects of FiftyOne.

Getting comfortable with using views and expressions to slice and dice your datasets based on the questions you have will allow you to work efficiently to curate high quality datasets.

- Creating Views
  - [Setup](#Setup)
  - [Overview](#Overview)
  - [View basics](#View-basics)
  - [View expressions](#View-expressions)
    - [View fields](#View-fields)
    - [Nested lists](#Nested-lists)
    - [Filtering list fields](#Filtering-list-fields)
    - [Mapping list fields](#Mapping-list-fields)
    - [Reducing list fields](#Reducing-list-fields)
    - [Referencing root fields](#Referencing-root-fields)
  - [Aggregations](#Aggregations)
  - [Summary](#Summary)