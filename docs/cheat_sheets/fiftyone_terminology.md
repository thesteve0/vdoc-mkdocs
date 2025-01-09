Table of Contents

- [Docs](../index.html) >

- [FiftyOne Cheat Sheets](index.html) >
- FiftyOne terminology

Contents


# FiftyOne terminology [¶](\#fiftyone-terminology "Permalink to this headline")

This cheat sheet introduces the key terminology in the world of FiftyOne.

## The basics [¶](\#the-basics "Permalink to this headline")

|     |     |
| --- | --- |
| FiftyOne | The [open-source framework](../index.html#fiftyone-library), the core library,<br>and the Python SDK. |
| FiftyOne App | The [provided user interface](../fiftyone_concepts/app.html#fiftyone-app) for graphically<br>viewing, filtering, and understanding your datasets. Can be launched in<br>the browser or within notebooks. |
| FiftyOne Teams | [The enterprise-grade suite](https://voxel51.com/fiftyone-teams/)<br>built on top of FiftyOne for collaboration, permissioning, and working<br>with cloud-backed media. |

## Other components [¶](\#other-components "Permalink to this headline")

|     |     |
| --- | --- |
| Brain | Library of [ML-powered capabilities](../brain.html#fiftyone-brain) for<br>computation and visualization. |
| Dataset Zoo | [Collection of common datasets](../dataset_zoo/index.html#dataset-zoo) available for<br>download and loading into FiftyOne. |
| Model Zoo | [Collection of pre-trained models](../data_and_models/model_zoo/index.html#model-zoo) available for<br>download and inference. |
| Plugin | A module you can use to [customize and extend](../plugins/index.html#fiftyone-plugins)<br>the behavior of FiftyOne. |
| Operator | A [plugin subcomponent](../plugins/using_plugins.html#using-operators) that defines an<br>operation that can be executed either directly by users in the App<br>and/or internally invoked by other plugin components |
| Integration | A dataset, ML framework, annotation service, or other tool FiftyOne is<br>[directly compatible with](../integrations/index.html#integrations). |

## Data model [¶](\#data-model "Permalink to this headline")

|     |     |
| --- | --- |
| Dataset | [Core data structure](../fiftyone_concepts/basics.html#basics-datasets) in FiftyOne, composed of<br>`Sample` instances. Provides a consistent interface for loading<br>images, videos, metadata, annotations, and predictions. The computer<br>vision analog of a table of data. |
| Sample | The atomic elements of a `Dataset` that store all the information<br>related to a given piece of data. Every [sample](../fiftyone_concepts/basics.html#basics-samples)<br>has an associated media file. The computer vision analog of a row of<br>tabular data. |
| DatasetView | [A view into](../fiftyone_concepts/using_views.html#using-views) a `Dataset`, which can filter,<br>sort, transform, etc. the dataset along various axes to obtain a<br>desired subset of the samples. |
| ViewStage | [A logical operation](../fiftyone_concepts/using_views.html#view-stages), such as filtering, matching,<br>or sorting, which can be used to generate a `DatasetView`. |
| Field | Attributes of `Sample` instances that<br>[store customizable information](../fiftyone_concepts/basics.html#basics-fields) about the<br>samples. The computer vision analog of a column in a table. |
| Embedded Document Field | [A collection of related fields](../fiftyone_concepts/using_datasets.html#custom-embedded-documents)<br>organized under a single top-level `Field`, similar to a nested<br>dictionary. |
| Label | Class hierarchy used to<br>[store semantic information](../fiftyone_concepts/basics.html#basics-labels) about ground truth<br>or predicted labels in a sample. Builtin `Label` types include<br>`Classification`, `Detections`, `Keypoints`, and many others. |
| Tag | A field containing a list of strings representing relevant<br>information. [Tags](../fiftyone_concepts/basics.html#basics-tags) can be assigned to datasets,<br>samples, or labels. |
| Metadata | A special `Sample` field that can be automatically populated with<br>media type-specific [metadata](../fiftyone_concepts/basics.html#basics-metadata) about the raw<br>media associated with the sample. |
| Aggregation | A class encapsulating the computation of an<br>[aggregate statistic](../fiftyone_concepts/basics.html#basics-aggregations) about the contents of<br>a dataset or view. |

## FiftyOne App [¶](\#fiftyone-app "Permalink to this headline")

|     |     |
| --- | --- |
| Session | [An instance of the FiftyOne App](../fiftyone_concepts/app.html#app-sessions) connected to a<br>specific dataset, via which you can use to programmatically interact<br>with the App. |
| Sample grid | The rectangular [media grid](../fiftyone_concepts/app.html#app-filtering) that you can scroll<br>through to quickly browse the samples in a dataset. Click on any media<br>in the grid to open the sample modal. |
| Sample modal | The [expanded modal](../fiftyone_concepts/app.html#app-sample-view) that provides detailed<br>information and visualization about an individual sample in a dataset. |
| Sidebar | Vertical component on [left side of App](../fiftyone_concepts/app.html#app-fields-sidebar)<br>that provides convenient options for filtering the dataset and<br>toggling the visibility of fields in the sample grid. |
| View bar | [Horizontal bar at the top of the App](../fiftyone_concepts/app.html#app-create-view) where<br>you can create and compose view stages via point and click operations<br>to filter your dataset and show only the content of interest. |

- FiftyOne terminology
  - [The basics](#the-basics)
  - [Other components](#other-components)
  - [Data model](#data-model)
  - [FiftyOne App](#fiftyone-app)