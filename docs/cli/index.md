Table of Contents

- [Docs](../index.html) >

- FiftyOne Command-Line Interface (CLI)

Contents


# FiftyOne Command-Line Interface (CLI) [¶](\#fiftyone-command-line-interface-cli "Permalink to this headline")

Installing FiftyOne automatically installs `fiftyone`, a command-line interface
(CLI) for interacting with FiftyOne. This utility provides access to many
useful features, including creating and inspecting datasets, visualizing
datasets in the App, exporting datasets and converting dataset formats,
and downloading datasets from the FiftyOne Dataset Zoo.

## Quickstart [¶](\#quickstart "Permalink to this headline")

To see the available top-level commands, type:

```
fiftyone --help

```

You can learn more about any available subcommand via:

```
fiftyone <command> --help

```

For example, to see your current FiftyOne config, you can execute
`fiftyone config`.

### Tab completion [¶](\#tab-completion "Permalink to this headline")

To enable tab completion in `bash`, add the following line to your `~/.bashrc`:

```
eval "$(register-python-argcomplete fiftyone)"

```

To enable tab completion in `zsh`, add these lines to your `~/.zshrc`:

```
autoload bashcompinit
bashcompinit
eval "$(register-python-argcomplete fiftyone)"

```

To enable tab completion in `tcsh`, add these lines to your `~/.tcshrc`:

```
eval `register-python-argcomplete --shell tcsh fiftyone`

```

## FiftyOne CLI [¶](\#cli-fiftyone-main "Permalink to this headline")

The FiftyOne command-line interface.

```
fiftyone [-h] [-v] [--all-help]
         {quickstart,annotation,brain,evaluation,app,config,constants,convert,datasets,migrate,operators,delegated,plugins,utils,zoo}
         ...

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show version info
  --all-help            show help recursively and exit

available commands:
  {quickstart,annotation,brain,evaluation,app,config,constants,convert,datasets,migrate,operators,delegated,plugins,utils,zoo}
    quickstart          Launch a FiftyOne quickstart.
    annotation          Tools for working with the FiftyOne annotation API.
    brain               Tools for working with the FiftyOne Brain.
    evaluation          Tools for working with the FiftyOne evaluation API.
    app                 Tools for working with the FiftyOne App.
    config              Tools for working with your FiftyOne config.
    constants           Print constants from `fiftyone.constants`.
    convert             Convert datasets on disk between supported formats.
    datasets            Tools for working with FiftyOne datasets.
    migrate             Tools for migrating the FiftyOne database.
    operators           Tools for working with FiftyOne operators.
    delegated           Tools for working with FiftyOne delegated operations.
    plugins             Tools for working with FiftyOne plugins.
    utils               FiftyOne utilities.
    zoo                 Tools for working with the FiftyOne Zoo.

```

## FiftyOne quickstart [¶](\#fiftyone-quickstart "Permalink to this headline")

Launch a FiftyOne quickstart.

```
fiftyone quickstart [-h] [-v] [-p PORT] [-A ADDRESS] [-r] [-a] [-w WAIT]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -v, --video           launch the quickstart with a video dataset
  -p PORT, --port PORT  the port number to use
  -A ADDRESS, --address ADDRESS
                        the address (server name) to use
  -r, --remote          whether to launch a remote App session
  -w WAIT, --wait WAIT  the number of seconds to wait for a new App
                        connection before returning if all connections are
                        lost. If negative, the process will wait forever,
                        regardless of connections

```

**Examples**

```
# Launch the quickstart
fiftyone quickstart

```

```
# Launch the quickstart with a video dataset
fiftyone quickstart --video

```

```
# Launch the quickstart as a remote session
fiftyone quickstart --remote

```

## FiftyOne config [¶](\#fiftyone-config "Permalink to this headline")

Tools for working with your FiftyOne config.

```
fiftyone config [-h] [-l] [FIELD]

```

**Arguments**

```
positional arguments:
  FIELD         a config field to print

optional arguments:
  -h, --help    show this help message and exit
  -l, --locate  print the location of your config on disk

```

**Examples**

```
# Print your entire config
fiftyone config

```

```
# Print a specific config field
fiftyone config <field>

```

```
# Print the location of your config on disk (if one exists)
fiftyone config --locate

```

## Print constants [¶](\#print-constants "Permalink to this headline")

Print constants from `fiftyone.constants`.

```
fiftyone constants [-h] [CONSTANT]

```

**Arguments**

```
positional arguments:
  CONSTANT    the constant to print

optional arguments:
  -h, --help  show this help message and exit

```

**Examples**

```
# Print all constants
fiftyone constants

```

```
# Print a specific constant
fiftyone constants <CONSTANT>

```

## Convert dataset formats [¶](\#convert-dataset-formats "Permalink to this headline")

Convert datasets on disk between supported formats.

```
fiftyone convert [-h] --input-type INPUT_TYPE --output-type OUTPUT_TYPE
                 [--input-dir INPUT_DIR]
                 [--input-kwargs KEY=VAL [KEY=VAL ...]]
                 [--output-dir OUTPUT_DIR]
                 [--output-kwargs KEY=VAL [KEY=VAL ...]] [-o]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
                        the directory containing the dataset
  --input-kwargs KEY=VAL [KEY=VAL ...]
                        additional keyword arguments for
                        `fiftyone.utils.data.convert_dataset(..., input_kwargs=)`
  --output-dir OUTPUT_DIR
                        the directory to which to write the output dataset
  --output-kwargs KEY=VAL [KEY=VAL ...]
                        additional keyword arguments for
                        `fiftyone.utils.data.convert_dataset(..., output_kwargs=)`
  -o, --overwrite       whether to overwrite an existing output directory

required arguments:
  --input-type INPUT_TYPE
                        the fiftyone.types.Dataset type of the input dataset
  --output-type OUTPUT_TYPE
                        the fiftyone.types.Dataset type to output

```

**Examples**

```
# Convert an image classification directory tree to TFRecords format
fiftyone convert \
    --input-dir /path/to/image-classification-directory-tree \
    --input-type fiftyone.types.ImageClassificationDirectoryTree \
    --output-dir /path/for/tf-image-classification-dataset \
    --output-type fiftyone.types.TFImageClassificationDataset

```

```
# Convert a COCO detection dataset to CVAT image format
fiftyone convert \
    --input-dir /path/to/coco-detection-dataset \
    --input-type fiftyone.types.COCODetectionDataset \
    --output-dir /path/for/cvat-image-dataset \
    --output-type fiftyone.types.CVATImageDataset

```

```
# Perform a customized conversion via optional kwargs
fiftyone convert \
    --input-dir /path/to/coco-detection-dataset \
    --input-type fiftyone.types.COCODetectionDataset \
    --input-kwargs max_samples=100 shuffle=True \
    --output-dir /path/for/cvat-image-dataset \
    --output-type fiftyone.types.TFObjectDetectionDataset \
    --output-kwargs force_rgb=True \
    --overwrite

```

## FiftyOne datasets [¶](\#fiftyone-datasets "Permalink to this headline")

Tools for working with FiftyOne datasets.

```
fiftyone datasets [-h] [--all-help]
                  {list,info,create,head,tail,stream,export,delete} ...

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recursively and exit

available commands:
  {list,info,create,head,tail,stream,export,delete}
    list                List FiftyOne datasets.
    info                Print information about FiftyOne datasets.
    stats               Print stats about FiftyOne datasets on disk.
    create              Tools for creating FiftyOne datasets.
    head                Prints the first few samples in a FiftyOne dataset.
    tail                Prints the last few samples in a FiftyOne dataset.
    stream              Streams the samples in a FiftyOne dataset.
    export              Export FiftyOne datasets to disk in supported formats.
    draw                Writes annotated versions of samples in FiftyOne datasets to disk.
    rename              Rename FiftyOne datasets.
    delete              Delete FiftyOne datasets.

```

### List datasets [¶](\#list-datasets "Permalink to this headline")

List FiftyOne datasets.

```
fiftyone datasets list [-h] [-p PATT] [-t TAG [TAG ...]]

```

**Arguments**

```
optional arguments:
  -h, --help        show this help message and exit
  -p PATT, --glob-patt PATT
                    an optional glob pattern of dataset names to include
  -t TAG [TAG ...], --tags TAG [TAG ...]
                    only show datasets with the given tag(s)

```

**Examples**

```
# List available datasets
fiftyone datasets list

```

```
# List datasets matching a given pattern
fiftyone datasets list --glob-patt 'quickstart-*'

```

```
# List datasets with the given tag(s)
fiftyone datasets list --tags automotive healthcare

```

### Print dataset information [¶](\#print-dataset-information "Permalink to this headline")

Print information about FiftyOne datasets.

```
fiftyone datasets info [-h] [-p PATT] [-t TAG [TAG ...]] [-s FIELD] [-r] [NAME]

```

**Arguments**

```
positional arguments:
  NAME                  the name of a dataset

optional arguments:
  -h, --help            show this help message and exit
  -p PATT, --glob-patt PATT
                        an optional glob pattern of dataset names to include
  -t TAG [TAG ...], --tags TAG [TAG ...]
                        only show datasets with the given tag(s)
  -s FIELD, --sort-by FIELD
                        a field to sort the dataset rows by
  -r, --reverse         whether to print the results in reverse order

```

**Examples**

```
# Print basic information about multiple datasets
fiftyone datasets info
fiftyone datasets info --glob-patt 'quickstart-*'
fiftyone datasets info --tags automotive healthcare
fiftyone datasets info --sort-by created_at
fiftyone datasets info --sort-by name --reverse

```

```
# Print information about a specific dataset
fiftyone datasets info <name>

```

### Print dataset stats [¶](\#print-dataset-stats "Permalink to this headline")

Print stats about FiftyOne datasets on disk.

```
fiftyone datasets stats [-h] [-m] [-c] NAME

```

**Arguments**

```
positional arguments:
  NAME                 the name of the dataset

optional arguments:
  -h, --help           show this help message and exit
  -m, --include-media  whether to include stats about the size of the raw
                       media in the dataset
  -c, --compressed     whether to return the sizes of collections in their
                       compressed form on disk

```

**Examples**

```
# Print stats about the given dataset on disk
fiftyone datasets stats <name>

```

### Create datasets [¶](\#create-datasets "Permalink to this headline")

Tools for creating FiftyOne datasets.

```
fiftyone datasets create [-h] [-n NAME] [-d DATASET_DIR] [-j JSON_PATH]
                         [-t TYPE] [-k KEY=VAL [KEY=VAL ...]]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  a name for the dataset
  -d DATASET_DIR, --dataset-dir DATASET_DIR
                        the directory containing the dataset
  -j JSON_PATH, --json-path JSON_PATH
                        the path to a samples JSON file to load
  -t TYPE, --type TYPE  the fiftyone.types.Dataset type of the dataset
  -k KEY=VAL [KEY=VAL ...], --kwargs KEY=VAL [KEY=VAL ...]
                        additional type-specific keyword arguments for
                        `fiftyone.core.dataset.Dataset.from_dir()`

```

**Examples**

```
# Create a dataset from the given data on disk
fiftyone datasets create \
    --name <name> --dataset-dir <dataset-dir> --type <type>

```

```
# Create a dataset from a random subset of the data on disk
fiftyone datasets create \
    --name <name> --dataset-dir <dataset-dir> --type <type> \
    --kwargs max_samples=50 shuffle=True

```

```
# Create a dataset from the given samples JSON file
fiftyone datasets create --json-path <json-path>

```

### Print dataset head [¶](\#print-dataset-head "Permalink to this headline")

Prints the first few samples in a FiftyOne dataset.

```
fiftyone datasets head [-h] [-n NUM_SAMPLES] NAME

```

**Arguments**

```
positional arguments:
  NAME                  the name of the dataset

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_SAMPLES, --num-samples NUM_SAMPLES
                        the number of samples to print

```

**Examples**

```
# Prints the first few samples in a dataset
fiftyone datasets head <name>

```

```
# Prints the given number of samples from the head of a dataset
fiftyone datasets head <name> --num-samples <num-samples>

```

### Print dataset tail [¶](\#print-dataset-tail "Permalink to this headline")

Prints the last few samples in a FiftyOne dataset.

```
fiftyone datasets tail [-h] [-n NUM_SAMPLES] NAME

```

**Arguments**

```
positional arguments:
  NAME                  the name of the dataset

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_SAMPLES, --num-samples NUM_SAMPLES
                        the number of samples to print

```

**Examples**

```
# Print the last few samples in a dataset
fiftyone datasets tail <name>

```

```
# Print the given number of samples from the tail of a dataset
fiftyone datasets tail <name> --num-samples <num-samples>

```

### Stream samples to the terminal [¶](\#stream-samples-to-the-terminal "Permalink to this headline")

Stream samples in a FiftyOne dataset to the terminal.

```
fiftyone datasets stream [-h] NAME

```

**Arguments**

```
positional arguments:
  NAME        the name of the dataset

optional arguments:
  -h, --help  show this help message and exit

```

**Examples**

```
# Stream the samples of the dataset to the terminal
fiftyone datasets stream <name>

```

### Export datasets [¶](\#export-datasets "Permalink to this headline")

Export FiftyOne datasets to disk in supported formats.

```
fiftyone datasets export [-h] [-d EXPORT_DIR] [-j JSON_PATH]
                         [-f LABEL_FIELD] [-t TYPE]
                         [--filters KEY=VAL [KEY=VAL ...]]
                         [-k KEY=VAL [KEY=VAL ...]]
                         NAME

```

**Arguments**

```
positional arguments:
  NAME                  the name of the dataset to export

optional arguments:
  -h, --help            show this help message and exit
  -d EXPORT_DIR, --export-dir EXPORT_DIR
                        the directory in which to export the dataset
  -j JSON_PATH, --json-path JSON_PATH
                        the path to export the dataset in JSON format
  -f LABEL_FIELD, --label-field LABEL_FIELD
                        the name of the label field to export
  -t TYPE, --type TYPE  the fiftyone.types.Dataset type in which to export
  --filters KEY=VAL [KEY=VAL ...]
                        specific sample tags or class labels to export. To
                        use sample tags, pass tags as `tags=train,val` and
                        to use label filters, pass label field and values
                        as in ground_truth=car,person,dog
  -k KEY=VAL [KEY=VAL ...], --kwargs KEY=VAL [KEY=VAL ...]
                        additional type-specific keyword arguments for
                        `fiftyone.core.collections.SampleCollection.export()`

```

**Examples**

```
# Export the dataset to disk in the specified format
fiftyone datasets export <name> \
    --export-dir <export-dir> --type <type> --label-field <label-field>

```

```
# Export the dataset to disk in JSON format
fiftyone datasets export <name> --json-path <json-path>

```

```
# Only export cats and dogs from the validation split
fiftyone datasets export <name> \\
    --filters tags=validation ground_truth=cat,dog \\
    --export-dir <export-dir> --type <type> --label-field ground_truth

```

```
# Perform a customized export of a dataset
fiftyone datasets export <name> \
    --type <type> \
    --kwargs labels_path=/path/for/labels.json

```

### Drawing labels on samples [¶](\#drawing-labels-on-samples "Permalink to this headline")

Renders annotated versions of samples in FiftyOne datasets to disk.

```
fiftyone datasets draw [-h] [-d OUTPUT_DIR] [-f LABEL_FIELDS] NAME

```

**Arguments**

```
positional arguments:
  NAME                  the name of the dataset

optional arguments:
  -h, --help            show this help message and exit
  -d OUTPUT_DIR, --output-dir OUTPUT_DIR
                        the directory to write the annotated media
  -f LABEL_FIELDS, --label-fields LABEL_FIELDS
                        a comma-separated list of label fields to export

```

**Examples**

```
# Write annotated versions of the media in the dataset with the
# specified label field(s) overlaid to disk
fiftyone datasets draw <name> \
    --output-dir <output-dir> --label-fields <list>,<of>,<fields>

```

### Rename datasets [¶](\#rename-datasets "Permalink to this headline")

Rename FiftyOne datasets.

```
fiftyone datasets rename [-h] NAME NEW_NAME

```

**Arguments**

```
positional arguments:
  NAME        the name of the dataset
  NEW_NAME    a new name for the dataset

optional arguments:
  -h, --help  show this help message and exit

```

**Examples**

```
# Rename the dataset
fiftyone datasets rename <old-name> <new-name>

```

### Delete datasets [¶](\#delete-datasets "Permalink to this headline")

Delete FiftyOne datasets.

```
fiftyone datasets delete [-h] [-g GLOB_PATT] [--non-persistent]
                         [NAME [NAME ...]]

```

**Arguments**

```
positional arguments:
  NAME                  the dataset name(s) to delete

optional arguments:
  -h, --help            show this help message and exit
  -g GLOB_PATT, --glob-patt GLOB_PATT
                        a glob pattern of datasets to delete
  --non-persistent      delete all non-persistent datasets

```

**Examples**

```
# Delete the datasets with the given name(s)
fiftyone datasets delete <name1> <name2> ...

```

```
# Delete the datasets whose names match the given glob pattern
fiftyone datasets delete --glob-patt <glob-patt>

```

```
# Delete all non-persistent datasets
fiftyone datasets delete --non-persistent

```

## FiftyOne migrations [¶](\#fiftyone-migrations "Permalink to this headline")

Tools for migrating the FiftyOne database.

See [this page](../fiftyone_concepts/config.html#database-migrations) for more information about migrating
FiftyOne deployments.

```
fiftyone migrate [-h] [-i] [-a]
                 [-v VERSION]
                 [-n DATASET_NAME [DATASET_NAME ...]]
                 [--error-level LEVEL]
                 [--verbose]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -i, --info            whether to print info about the current revisions
  -a, --all             whether to migrate the database and all datasets
  -v VERSION, --version VERSION
                        the revision to migrate to
  -n DATASET_NAME [DATASET_NAME ...], --dataset-name DATASET_NAME [DATASET_NAME ...]
                        the name of a specific dataset to migrate
  --error-level LEVEL   the error level (0=error, 1=warn, 2=ignore) to use
                        when migrating individual datasets
  --verbose             whether to log incremental migrations that are performed

```

**Examples**

```
# Print information about the current revisions of all datasets
fiftyone migrate --info

```

```
# Migrate the database and all datasets to the current client version
fiftyone migrate --all

```

```
# Migrate to a specific revision
fiftyone migrate --all --version <VERSION>

```

```
# Migrate a specific dataset
fiftyone migrate ... --dataset-name <DATASET_NAME>

```

```
# Update the database version without migrating any existing datasets
fiftyone migrate

```

## FiftyOne operators [¶](\#fiftyone-operators "Permalink to this headline")

Tools for working with FiftyOne operators and panels.

```
fiftyone operators [-h] [--all-help] {list,info} ...

```

**Arguments**

```
optional arguments:
  -h, --help   show this help message and exit
  --all-help   show help recursively and exit

available commands:
  {list,info}
    list       List operators and panels that you've installed locally.
    info       Prints information about operators and panels that you've installed locally.

```

### List operators [¶](\#list-operators "Permalink to this headline")

List operators and panels that you’ve installed locally.

```
fiftyone operators list [-h] [-e] [-d] [-o] [-p] [-n]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -e, --enabled         only show enabled operators and panels
  -d, --disabled        only show disabled operators and panels
  -o, --operators-only  only show operators
  -p, --panels-only     only show panels
  -n, --names-only      only show names

```

**Examples**

```
# List all available operators and panels
fiftyone operators list

```

```
# List enabled operators and panels
fiftyone operators list --enabled

```

```
# List disabled operators and panels
fiftyone operators list --disabled

```

```
# Only list panels
fiftyone operators list --panels-only

```

### Operator info [¶](\#operator-info "Permalink to this headline")

Prints information about operators and panels that you’ve installed locally.

```
fiftyone operators info [-h] URI

```

**Arguments**

```
positional arguments:
  URI         the operator or panel URI

optional arguments:
  -h, --help  show this help message and exit

```

**Examples**

```
# Prints information about an operator or panel
fiftyone operators info <uri>

```

## FiftyOne delegated operations [¶](\#fiftyone-delegated-operations "Permalink to this headline")

Tools for working with FiftyOne delegated operations.

```
fiftyone delegated [-h] [--all-help] {launch,list,info,fail,delete,cleanup} ...

```

**Arguments**

```
optional arguments:
  -h, --help   show this help message and exit
  --all-help   show help recursively and exit

available commands:
  {launch,list,info,fail,delete,cleanup}
    launch              Launches a service for running delegated operations.
    list                List delegated operations.
    info                Prints information about a delegated operation.
    fail                Manually mark delegated as failed.
    delete              Delete delegated operations.
    cleanup             Cleanup delegated operations.

```

### Launch delegated service [¶](\#launch-delegated-service "Permalink to this headline")

Launches a service for running delegated operations.

```
fiftyone delegated launch [-h] [-t TYPE]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -t TYPE, --type TYPE  the type of service to launch. The default is 'local'

```

**Examples**

```
# Launch a local service
fiftyone delegated launch

```

### List delegated operations [¶](\#list-delegated-operations "Permalink to this headline")

List delegated operations.

```
fiftyone delegated list [-h]
                        [-o OPERATOR]
                        [-d DATASET]
                        [-s STATE]
                        [--sort-by SORT_BY]
                        [--reverse]
                        [-l LIMIT]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -o OPERATOR, --operator OPERATOR
                        only list operations for this operator
  -d DATASET, --dataset DATASET
                        only list operations for this dataset
  -s STATE, --state STATE
                        only list operations with this state. Supported
                        values are ('SCHEDULED', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED')
  --sort-by SORT_BY     how to sort the operations. Supported values are
                        ('SCHEDULED_AT', 'QUEUED_AT', 'STARTED_AT', COMPLETED_AT', 'FAILED_AT', 'OPERATOR')
  --reverse             whether to sort in reverse order
  -l LIMIT, --limit LIMIT
                        a maximum number of operations to show

```

**Examples**

```
# List all delegated operations
fiftyone delegated list

```

```
# List some specific delegated operations
fiftyone delegated list \
    --dataset quickstart \
    --operator @voxel51/io/export_samples \
    --state COMPLETED \
    --sort-by COMPLETED_AT \
    --limit 10

```

### Delegated operation info [¶](\#delegated-operation-info "Permalink to this headline")

Prints information about a delegated operation.

```
fiftyone delegated info [-h] ID

```

**Arguments**

```
positional arguments:
  ID          the operation ID

optional arguments:
  -h, --help  show this help message and exit

```

**Examples**

```
# Print information about a delegated operation
fiftyone delegated info <id>

```

### Mark delegated operations as failed [¶](\#mark-delegated-operations-as-failed "Permalink to this headline")

Manually mark delegated operations as failed.

```
fiftyone delegated fail [-h] [IDS ...]

```

**Arguments**

```
positional arguments:
  IDS         an operation ID or list of operation IDs

optional arguments:
  -h, --help  show this help message and exit

```

**Examples**

```
# Manually mark the specified operation(s) as FAILED
fiftyone delegated fail <id1> <id2> ...

```

### Delete delegated operations [¶](\#delete-delegated-operations "Permalink to this headline")

Delete delegated operations.

```
fiftyone delegated delete [-h] [IDS ...]

```

**Arguments**

```
positional arguments:
  IDS         an operation ID or list of operation IDs

optional arguments:
  -h, --help  show this help message and exit

```

**Examples**

```
# Delete the specified operation(s)
fiftyone delegated delete <id1> <id2> ...

```

### Cleanup delegated operations [¶](\#cleanup-delegated-operations "Permalink to this headline")

Cleanup delegated operations.

```
fiftyone delegated cleanup [-h]
                           [-o OPERATOR]
                           [-d DATASET]
                           [-s STATE]
                           [--orphan]
                           [--dry-run]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -o OPERATOR, --operator OPERATOR
                        cleanup operations for this operator
  -d DATASET, --dataset DATASET
                        cleanup operations for this dataset
  -s STATE, --state STATE
                        delete operations in this state. Supported values
                        are ('QUEUED', 'COMPLETED', 'FAILED')
  --orphan              delete all operations associated with non-existent
                        datasets
  --dry-run             whether to print information rather than actually
                        deleting operations

```

**Examples**

```
# Delete all failed operations associated with a given dataset
fiftyone delegated cleanup --dataset quickstart --state FAILED

```

```
# Delete all delegated operations associated with non-existent datasets
fiftyone delegated cleanup --orphan

```

```
# Print information about operations rather than actually deleting them
fiftyone delegated cleanup --orphan --dry-run

```

## FiftyOne plugins [¶](\#fiftyone-plugins "Permalink to this headline")

Tools for working with FiftyOne plugins.

```
fiftyone plugins [-h] [--all-help] {list,info,download,requirements,create,enable,disable,delete} ...

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recursively and exit

available commands:
  {list,info,download,requirements,create,enable,disable,delete}
    list                List plugins that you've downloaded or created locally.
    info                Prints information about plugins that you've downloaded or created
    download            Download plugins from the web.
    requirements        Handles package requirements for plugins.
    create              Creates or initializes a plugin.
    enable              Enables the given plugin(s).
    disable             Disables the given plugin(s).
    delete              Delete plugins from your local machine.

```

### List plugins [¶](\#list-plugins "Permalink to this headline")

List plugins that you’ve downloaded or created locally.

```
fiftyone plugins list [-h] [-e] [-d] [-n]

```

**Arguments**

```
optional arguments:
  -h, --help        show this help message and exit
  -e, --enabled     only show enabled plugins
  -d, --disabled    only show disabled plugins
  -n, --names-only  only show plugin names

```

**Examples**

```
# List all locally available plugins
fiftyone plugins list

```

```
# List enabled plugins
fiftyone plugins list --enabled

```

```
# List disabled plugins
fiftyone plugins list --disabled

```

### Plugin info [¶](\#plugin-info "Permalink to this headline")

List plugins that you’ve downloaded or created locally.

```
fiftyone plugins info [-h] NAME

```

**Arguments**

```
positional arguments:
  NAME        the plugin name

optional arguments:
  -h, --help  show this help message and exit

```

**Examples**

```
# Prints information about a plugin
fiftyone plugins info <name>

```

### Download plugins [¶](\#download-plugins "Permalink to this headline")

Download plugins from the web.

When downloading plugins from GitHub, you can provide any of the following
formats:

- a GitHub repo URL like `https://github.com/<user>/<repo>`

- a GitHub ref like `https://github.com/<user>/<repo>/tree/<branch>` or
`https://github.com/<user>/<repo>/commit/<commit>`

- a GitHub ref string like `<user>/<repo>[/<ref>]`


Note

To download from a private GitHub repository that you have access to,
provide your GitHub personal access token by setting the `GITHUB_TOKEN`
environment variable.

```
fiftyone plugins download [-h] [-n [PLUGIN_NAMES ...]] [-o] URL_OR_GH_REPO

```

**Arguments**

```
positional arguments:
  URL_OR_GH_REPO        A URL or <user>/<repo>[/<ref>] of a GitHub repository

optional arguments:
  -h, --help            show this help message and exit
  -n [PLUGIN_NAMES ...], --plugin-names [PLUGIN_NAMES ...]
                        a plugin name or list of plugin names to download
  -o, --overwrite       whether to overwrite existing plugins

```

**Examples**

```
# Download plugins from a GitHub repository URL
fiftyone plugins download <github-repo-url>

```

```
# Download plugins by specifying the GitHub repository details
fiftyone plugins download <user>/<repo>[/<ref>]

```

```
# Download specific plugins from a URL
fiftyone plugins download <url> --plugin-names <name1> <name2> <name3>

```

### Plugin requirements [¶](\#plugin-requirements "Permalink to this headline")

Handles package requirements for plugins.

```
fiftyone plugins requirements [-h] [-p] [-i] [-e] [--error-level LEVEL] NAME

```

**Arguments**

```
positional arguments:
  NAME                 the plugin name

optional arguments:
  -h, --help           show this help message and exit
  -p, --print          print the requirements for the plugin
  -i, --install        install any requirements for the plugin
  -e, --ensure         ensure the requirements for the plugin are satisfied
  --error-level LEVEL  the error level (0=error, 1=warn, 2=ignore) to use when installing or ensuring plugin requirements

```

**Examples**

```
# Print requirements for a plugin
fiftyone plugins requirements <name> --print

```

```
# Install any requirements for the plugin
fiftyone plugins requirements <name> --install

```

```
# Ensures that the requirements for the plugin are satisfied
fiftyone plugins requirements <name> --ensure

```

### Create plugins [¶](\#create-plugins "Permalink to this headline")

Creates or initializes a plugin.

```
fiftyone plugins create [-h]
                        [-f [FILES ...]]
                        [-d OUTDIR]
                        [--label LABEL]
                        [--description DESCRIPTION]
                        [--version VERSION]
                        [-o]
                        [--kwargs KEY=VAL [KEY=VAL ...]]
                        [NAME ...]

```

**Arguments**

```
positional arguments:
  NAME                  the plugin name(s)

optional arguments:
  -h, --help            show this help message and exit
  -f [FILES ...], --from-files [FILES ...]
                        a directory or list of explicit filepaths to include in the plugin
  -d OUTDIR, --outdir OUTDIR
                        a directory in which to create the plugin
  --label LABEL         a display name for the plugin
  --description DESCRIPTION
                        a description for the plugin
  --version VERSION     an optional FiftyOne version requirement for the plugin
  -o, --overwrite       whether to overwrite existing plugins
  --kwargs KEY=VAL [KEY=VAL ...]
                        additional keyword arguments to include in the plugin definition

```

**Examples**

```
# Initialize a new plugin
fiftyone plugins create <name>

```

```
# Create a plugin from existing files
fiftyone plugins create \
    <name> \
    --from-files /path/to/dir \
    --label <label> \
    --description <description>

```

### Enable plugins [¶](\#enable-plugins "Permalink to this headline")

Enables the given plugin(s).

```
fiftyone plugins enable [-h] [-a] [NAME ...]

```

**Arguments**

```
positional arguments:
  NAME        the plugin name(s)

optional arguments:
  -h, --help  show this help message and exit
  -a, --all   whether to enable all plugins

```

**Examples**

```
# Enable a plugin
fiftyone plugins enable <name>

```

```
# Enable multiple plugins
fiftyone plugins enable <name1> <name2> ...

```

```
# Enable all plugins
fiftyone plugins enable --all

```

### Disable plugins [¶](\#disable-plugins "Permalink to this headline")

Disables the given plugin(s).

```
fiftyone plugins disable [-h] [-a] [NAME ...]

```

**Arguments**

```
positional arguments:
  NAME        the plugin name(s)

optional arguments:
  -h, --help  show this help message and exit
  -a, --all   whether to disable all plugins

```

**Examples**

```
# Disable a plugin
fiftyone plugins disable <name>

```

```
# Disable multiple plugins
fiftyone plugins disable <name1> <name2> ...

```

```
# Disable all plugins
fiftyone plugins disable --all

```

### Delete plugins [¶](\#delete-plugins "Permalink to this headline")

Delete plugins from your local machine.

```
fiftyone plugins delete [-h] [-a] [NAME ...]

```

**Arguments**

```
positional arguments:
  NAME        the plugin name(s)

optional arguments:
  -h, --help  show this help message and exit
  -a, --all   whether to delete all plugins

```

**Examples**

```
# Delete a plugin from local disk
fiftyone plugins delete <name>

```

```
# Delete multiple plugins from local disk
fiftyone plugins delete <name1> <name2> ...

```

```
# Delete all plugins from local disk
fiftyone plugins delete --all

```

## FiftyOne utilities [¶](\#fiftyone-utilities "Permalink to this headline")

FiftyOne utilities.

```
fiftyone utils [-h] [--all-help]
               {compute-metadata,transform-images,transform-videos} ...

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recursively and exit

available commands:
  {compute-metadata,transform-images,transform-videos}
    compute-metadata    Populates the `metadata` field of all samples in the dataset.
    transform-images    Transforms the images in a dataset per the specified parameters.
    transform-videos    Transforms the videos in a dataset per the specified parameters.

```

### Compute metadata [¶](\#compute-metadata "Permalink to this headline")

Populates the `metadata` field of all samples in the dataset.

```
fiftyone utils compute-metadata [-h] [-o] [-n NUM_WORKERS] [-s] DATASET_NAME

```

**Arguments**

```
positional arguments:
  NAME                  the name of the dataset

optional arguments:
  -h, --help            show this help message and exit
  -o, --overwrite       whether to overwrite existing metadata
  -n NUM_WORKERS, --num-workers NUM_WORKERS
                        a suggested number of worker processes to use
  -s, --skip-failures   whether to gracefully continue without raising an
                        error if metadata cannot be computed for a sample

```

**Examples**

```
# Populate all missing `metadata` sample fields
fiftyone utils compute-metadata <dataset-name>

```

```
# (Re)-populate the `metadata` field for all samples
fiftyone utils compute-metadata <dataset-name> --overwrite

```

### Transform images [¶](\#transform-images "Permalink to this headline")

Transforms the images in a dataset per the specified parameters.

```
fiftyone utils transform-images [-h] [--size SIZE] [--min-size MIN_SIZE]
                                [--max-size MAX_SIZE] [-i INTERPOLATION]
                                [-e EXT] [-f] [--media-field MEDIA_FIELD]
                                [--output-field OUTPUT_FIELD]
                                [-o OUTPUT_DIR] [-r REL_DIR]
                                [--no-update-filepaths]
                                [-d] [-n NUM_WORKERS] [-s]
                                DATASET_NAME

```

**Arguments**

```
positional arguments:
  DATASET_NAME          the name of the dataset

optional arguments:
  -h, --help            show this help message and exit
  --size SIZE           a `width,height` for each image. A dimension can be
                        -1 if no constraint should be applied
  --min-size MIN_SIZE   a minimum `width,height` for each image. A dimension
                        can be -1 if no constraint should be applied
  --max-size MAX_SIZE   a maximum `width,height` for each image. A dimension
                        can be -1 if no constraint should be applied
  -i INTERPOLATION, --interpolation INTERPOLATION
                        an optional `interpolation` argument for `cv2.resize()`
  -e EXT, --ext EXT     an image format to convert to (e.g., '.png' or '.jpg')
  -f, --force-reencode  whether to re-encode images whose parameters already
                        meet the specified values
  --media-field MEDIA_FIELD
                        the input field containing the image paths to
                        transform
  --output-field OUTPUT_FIELD
                        an optional field in which to store the paths to
                        the transformed images. By default, `media_field`
                        is updated in-place
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        an optional output directory in which to write the
                        transformed images. If none is provided, the images
                        are updated in-place
  -r REL_DIR, --rel-dir REL_DIR
                        an optional relative directory to strip from each
                        input filepath to generate a unique identifier that
                        is joined with `output_dir` to generate an output
                        path for each image
  --no-update-filepaths
                        whether to store the output filepaths on the sample
                        collection
  -d, --delete-originals
                        whether to delete the original images after transforming
  -n NUM_WORKERS, --num-workers NUM_WORKERS
                        a suggested number of worker processes to use
  -s, --skip-failures   whether to gracefully continue without raising an
                        error if an image cannot be transformed

```

**Examples**

```
# Convert the images in the dataset to PNGs
fiftyone utils transform-images <dataset-name> --ext .png --delete-originals

```

```
# Ensure that no images in the dataset exceed 1920 x 1080
fiftyone utils transform-images <dataset-name> --max-size 1920,1080

```

### Transform videos [¶](\#transform-videos "Permalink to this headline")

Transforms the videos in a dataset per the specified parameters.

```
fiftyone utils transform-videos [-h] [--fps FPS] [--min-fps MIN_FPS]
                                [--max-fps MAX_FPS] [--size SIZE]
                                [--min-size MIN_SIZE] [--max-size MAX_SIZE]
                                [-r] [-f]
                                [--media-field MEDIA_FIELD]
                                [--output-field OUTPUT_FIELD]
                                [--output-dir OUTPUT_DIR]
                                [--rel-dir REL_DIR]
                                [--no-update-filepaths]
                                [-d] [-s] [-v]
                                DATASET_NAME

```

**Arguments**

```
positional arguments:
  DATASET_NAME          the name of the dataset

optional arguments:
  -h, --help            show this help message and exit
  --fps FPS             a frame rate at which to resample the videos
  --min-fps MIN_FPS     a minimum frame rate. Videos with frame rate below
                        this value are upsampled
  --max-fps MAX_FPS     a maximum frame rate. Videos with frame rate exceeding
                        this value are downsampled
  --size SIZE           a `width,height` for each frame. A dimension can be -1
                        if no constraint should be applied
  --min-size MIN_SIZE   a minimum `width,height` for each frame. A dimension
                        can be -1 if no constraint should be applied
  --max-size MAX_SIZE   a maximum `width,height` for each frame. A dimension
                        can be -1 if no constraint should be applied
  -r, --reencode        whether to re-encode the videos as H.264 MP4s
  -f, --force-reencode  whether to re-encode videos whose parameters already
                        meet the specified values
  --media-field MEDIA_FIELD
                        the input field containing the video paths to
                        transform
  --output-field OUTPUT_FIELD
                        an optional field in which to store the paths to
                        the transformed videos. By default, `media_field`
                        is updated in-place
  --output-dir OUTPUT_DIR
                        an optional output directory in which to write the
                        transformed videos. If none is provided, the videos
                        are updated in-place
  --rel-dir REL_DIR     an optional relative directory to strip from each
                        input filepath to generate a unique identifier that
                        is joined with `output_dir` to generate an output
                        path for each video
  --no-update-filepaths
                        whether to store the output filepaths on the sample
                        collection
  -d, --delete-originals
                        whether to delete the original videos after transforming
  -s, --skip-failures   whether to gracefully continue without raising an
                        error if a video cannot be transformed
  -v, --verbose         whether to log the `ffmpeg` commands that are executed

```

**Examples**

```
# Re-encode the videos in the dataset as H.264 MP4s
fiftyone utils transform-videos <dataset-name> --reencode

```

```
# Ensure that no videos in the dataset exceed 1920 x 1080 and 30fps
fiftyone utils transform-videos <dataset-name> \
    --max-size 1920,1080 --max-fps 30.0

```

## FiftyOne Annotation [¶](\#fiftyone-annotation "Permalink to this headline")

Tools for working with the FiftyOne annotation API.

```
fiftyone annotation [-h] [--all-help] {config} ...

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recursively and exit

available commands:
  {config}
    config              Tools for working with your FiftyOne annotation config.

```

### Annotation Config [¶](\#annotation-config "Permalink to this headline")

Tools for working with your FiftyOne annotation config.

```
fiftyone annotation config [-h] [-l] [FIELD]

```

**Arguments**

```
positional arguments:
  FIELD         an annotation config field to print

optional arguments:
  -h, --help    show this help message and exit
  -l, --locate  print the location of your annotation config on disk

```

**Examples**

```
# Print your entire annotation config
fiftyone annotation config

```

```
# Print a specific annotation config field
fiftyone annotation config <field>

```

```
# Print the location of your annotation config on disk (if one exists)
fiftyone annotation config --locate

```

## FiftyOne App [¶](\#fiftyone-app "Permalink to this headline")

Tools for working with the FiftyOne App.

```
fiftyone app [-h] [--all-help] {config,launch,view,connect} ...

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recursively and exit

available commands:
  {config,launch,view,connect}
    config              Tools for working with your App config.
    launch              Launch the FiftyOne App.
    view                View datasets in the App without persisting them to the database.
    connect             Connect to a remote FiftyOne App.

```

### App Config [¶](\#app-config "Permalink to this headline")

Tools for working with your FiftyOne App config.

```
fiftyone app config [-h] [-l] [FIELD]

```

**Arguments**

```
positional arguments:
  FIELD         an App config field to print

optional arguments:
  -h, --help    show this help message and exit
  -l, --locate  print the location of your App config on disk

```

**Examples**

```
# Print your entire App config
fiftyone app config

```

```
# Print a specific App config field
fiftyone app config <field>

```

```
# Print the location of your App config on disk (if one exists)
fiftyone app config --locate

```

### Launch the App [¶](\#launch-the-app "Permalink to this headline")

Launch the FiftyOne App.

```
fiftyone app launch [-h] [-p PORT] [-A ADDRESS] [-b BROWSER] [-r] [-a] [-w WAIT] [NAME]

```

**Arguments**

```
positional arguments:
  NAME                  the name of a dataset to open

optional arguments:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  the port number to use
  -A ADDRESS, --address ADDRESS
                        the address (server name) to use
  -r, --remote          whether to launch a remote App session
  -b BROWSER, --browser BROWSER
                        the browser to use to open the App
  -w WAIT, --wait WAIT  the number of seconds to wait for a new App
                        connection before returning if all connections are
                        lost. If negative, the process will wait forever,
                        regardless of connections

```

**Examples**

```
# Launch the App
fiftyone app launch

```

```
# Launch the App with the given dataset loaded
fiftyone app launch <name>

```

```
# Launch a remote App session
fiftyone app launch ... --remote

```

```
# Launch an App session with a specific browser
fiftyone app launch ... --browser <name>

```

### View datasets in App [¶](\#view-datasets-in-app "Permalink to this headline")

View datasets in the FiftyOne App without persisting them to the database.

```
fiftyone app view [-h] [-n NAME] [-d DATASET_DIR] [-t TYPE] [-z NAME]
                  [-s SPLITS [SPLITS ...]] [--images-dir IMAGES_DIR]
                  [--images-patt IMAGES_PATT] [--videos-dir VIDEOS_DIR]
                  [--videos-patt VIDEOS_PATT] [-j JSON_PATH] [-p PORT]
                  [-A ADDRESS] [-r] [-a] [-w WAIT]
                  [-k KEY=VAL [KEY=VAL ...]]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  a name for the dataset
  -d DATASET_DIR, --dataset-dir DATASET_DIR
                        the directory containing the dataset to view
  -t TYPE, --type TYPE  the fiftyone.types.Dataset type of the dataset
  -z NAME, --zoo-dataset NAME
                        the name of a zoo dataset to view
  -s SPLITS [SPLITS ...], --splits SPLITS [SPLITS ...]
                        the dataset splits to load
  --images-dir IMAGES_DIR
                        the path to a directory of images
  --images-patt IMAGES_PATT
                        a glob pattern of images
  --videos-dir VIDEOS_DIR
                        the path to a directory of videos
  --videos-patt VIDEOS_PATT
                        a glob pattern of videos
  -j JSON_PATH, --json-path JSON_PATH
                        the path to a samples JSON file to view
  -p PORT, --port PORT  the port number to use
  -A ADDRESS, --address ADDRESS
                        the address (server name) to use
  -r, --remote          whether to launch a remote App session
  -w WAIT, --wait WAIT  the number of seconds to wait for a new App
                        connection before returning if all connections are
                        lost. If negative, the process will wait forever,
                        regardless of connections
  -k KEY=VAL [KEY=VAL ...], --kwargs KEY=VAL [KEY=VAL ...]
                        additional type-specific keyword arguments for
                        `fiftyone.core.dataset.Dataset.from_dir()`

```

**Examples**

```
# View a dataset stored on disk in the App
fiftyone app view --dataset-dir <dataset-dir> --type <type>

```

```
# View a zoo dataset in the App
fiftyone app view --zoo-dataset <name> --splits <split1> ...

```

```
# View a directory of images in the App
fiftyone app view --images-dir <images-dir>

```

```
# View a glob pattern of images in the App
fiftyone app view --images-patt <images-patt>

```

```
# View a directory of videos in the App
fiftyone app view --videos-dir <videos-dir>

```

```
# View a glob pattern of videos in the App
fiftyone app view --videos-patt <videos-patt>

```

```
# View a dataset stored in JSON format on disk in the App
fiftyone app view --json-path <json-path>

```

```
# View the dataset in a remote App session
fiftyone app view ... --remote

```

```
# View a random subset of the data stored on disk in the App
fiftyone app view ... --kwargs max_samples=50 shuffle=True

```

### Connect to remote App [¶](\#connect-to-remote-app "Permalink to this headline")

Connect to a remote FiftyOne App in your web browser.

```
fiftyone app connect [-h] [-d DESTINATION] [-p PORT] [-A ADDRESS] [-l PORT]
                     [-i KEY]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -d DESTINATION, --destination DESTINATION
                        the destination to connect to, e.g., [username@]hostname
  -p PORT, --port PORT  the remote port to connect to
  -l PORT, --local-port PORT
                        the local port to use to serve the App
  -i KEY, --ssh-key KEY
                        optional ssh key to use to login

```

**Examples**

```
# Connect to a remote App with port forwarding already configured
fiftyone app connect

```

```
# Connect to a remote App session
fiftyone app connect --destination <destination> --port <port>

```

```
# Connect to a remote App session using an ssh key
fiftyone app connect ... --ssh-key <path/to/key>

```

```
# Connect to a remote App using a custom local port
fiftyone app connect ... --local-port <port>

```

## FiftyOne Brain [¶](\#fiftyone-brain "Permalink to this headline")

Tools for working with the FiftyOne Brain.

```
fiftyone brain [-h] [--all-help] {config} ...

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recursively and exit

available commands:
  {config}
    config              Tools for working with your FiftyOne Brain config.

```

### Brain Config [¶](\#brain-config "Permalink to this headline")

Tools for working with your FiftyOne Brain config.

```
fiftyone brain config [-h] [-l] [FIELD]

```

**Arguments**

```
positional arguments:
  FIELD         a brain config field to print

optional arguments:
  -h, --help    show this help message and exit
  -l, --locate  print the location of your brain config on disk

```

**Examples**

```
# Print your entire brain config
fiftyone brain config

```

```
# Print a specific brain config field
fiftyone brain config <field>

```

```
# Print the location of your brain config on disk (if one exists)
fiftyone brain config --locate

```

## FiftyOne Evaluation [¶](\#fiftyone-evaluation "Permalink to this headline")

Tools for working with the FiftyOne evaluation API.

```
fiftyone evaluation [-h] [--all-help] {config} ...

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recursively and exit

available commands:
  {config}
    config              Tools for working with your FiftyOne evaluation config.

```

### Evaluation Config [¶](\#evaluation-config "Permalink to this headline")

Tools for working with your FiftyOne evaluation config.

```
fiftyone evaluation config [-h] [-l] [FIELD]

```

**Arguments**

```
positional arguments:
  FIELD         an evaluation config field to print

optional arguments:
  -h, --help    show this help message and exit
  -l, --locate  print the location of your evaluation config on disk

```

**Examples**

```
# Print your entire evaluation config
fiftyone evaluation config

```

```
# Print a specific evaluation config field
fiftyone evaluation config <field>

```

```
# Print the location of your evaluation config on disk (if one exists)
fiftyone evaluation config --locate

```

## FiftyOne Zoo [¶](\#fiftyone-zoo "Permalink to this headline")

Tools for working with the FiftyOne Zoo.

```
fiftyone zoo [-h] [--all-help] {datasets,models} ...

```

**Arguments**

```
optional arguments:
  -h, --help         show this help message and exit
  --all-help         show help recursively and exit

available commands:
  {datasets,models}
    datasets         Tools for working with the FiftyOne Dataset Zoo.
    models           Tools for working with the FiftyOne Model Zoo.

```

## FiftyOne Dataset Zoo [¶](\#fiftyone-dataset-zoo "Permalink to this headline")

Tools for working with the FiftyOne Dataset Zoo.

```
fiftyone zoo datasets [-h] [--all-help]
                      {list,find,info,download,load,delete} ...

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recursively and exit

available commands:
  {list,find,info,download,load,delete}
    list                List datasets in the FiftyOne Dataset Zoo.
    find                Locate a downloaded zoo dataset on disk.
    info                Print information about datasets in the FiftyOne Dataset Zoo.
    download            Download zoo datasets.
    load                Load zoo datasets as persistent FiftyOne datasets.
    delete              Deletes the local copy of the zoo dataset on disk.

```

### List datasets in zoo [¶](\#list-datasets-in-zoo "Permalink to this headline")

List datasets in the FiftyOne Dataset Zoo.

```
fiftyone zoo datasets list [-h] [-n] [-d] [-s SOURCE] [-t TAGS]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -n, --names-only      only show dataset names
  -d, --downloaded-only
                        only show datasets that have been downloaded
  -s SOURCE, --source SOURCE
                        only show datasets available from the specified source
  -t TAGS, --tags TAGS  only show datasets with the specified tag or list,of,tags

```

**Examples**

```
# List available datasets
fiftyone zoo datasets list

```

```
# List available dataset names
fiftyone zoo datasets list --names-only

```

```
# List downloaded datasets
fiftyone zoo datasets list --downloaded-only

```

```
# List available datasets from the given source
fiftyone zoo datasets list --source <source>

```

```
# List available datasets with the given tag
fiftyone zoo datasets list --tags <tag>

```

### Find zoo datasets on disk [¶](\#find-zoo-datasets-on-disk "Permalink to this headline")

Locate a downloaded zoo dataset on disk.

```
fiftyone zoo datasets find [-h] [-s SPLIT] NAME_OR_URL

```

**Arguments**

```
positional arguments:
  NAME_OR_URL           the name or remote location of the dataset

optional arguments:
  -h, --help            show this help message and exit
  -s SPLIT, --split SPLIT

```

**Examples**

```
# Print the location of a downloaded zoo dataset on disk
fiftyone zoo datasets find <name>

```

```
# Print the location of a remotely-sourced zoo dataset on disk
fiftyone zoo datasets find https://github.com/<user>/<repo>
fiftyone zoo datasets find <url>

```

```
# Print the location of a specific split of a dataset
fiftyone zoo datasets find <name> --split <split>

```

### Show zoo dataset info [¶](\#show-zoo-dataset-info "Permalink to this headline")

Print information about datasets in the FiftyOne Dataset Zoo.

```
fiftyone zoo datasets info [-h] NAME_OR_URL

```

**Arguments**

```
positional arguments:
  NAME_OR_URL           the name or remote location of the dataset

optional arguments:
  -h, --help            show this help message and exit

```

**Examples**

```
# Print information about a zoo dataset
fiftyone zoo datasets info <name>

```

```
# Print information about a remote zoo dataset
fiftyone zoo datasets info https://github.com/<user>/<repo>
fiftyone zoo datasets info <url>

```

### Download zoo datasets [¶](\#download-zoo-datasets "Permalink to this headline")

Download zoo datasets.

When downloading remotely-sourced zoo datasets, you can provide any of the
following formats:

- a GitHub repo URL like `https://github.com/<user>/<repo>`

- a GitHub ref like `https://github.com/<user>/<repo>/tree/<branch>` or
`https://github.com/<user>/<repo>/commit/<commit>`

- a GitHub ref string like `<user>/<repo>[/<ref>]`

- a publicly accessible URL of an archive (eg zip or tar) file


Note

To download from a private GitHub repository that you have access to,
provide your GitHub personal access token by setting the `GITHUB_TOKEN`
environment variable.

```
fiftyone zoo datasets download [-h] [-s SPLITS [SPLITS ...]]
                               [-k KEY=VAL [KEY=VAL ...]]
                               NAME_OR_URL

```

**Arguments**

```
positional arguments:
  NAME_OR_URL           the name or remote location of the dataset

optional arguments:

  -h, --help            show this help message and exit
  -s SPLITS [SPLITS ...], --splits SPLITS [SPLITS ...]
                        the dataset splits to download
  -k KEY=VAL [KEY=VAL ...], --kwargs KEY=VAL [KEY=VAL ...]
                        optional dataset-specific keyword arguments for
                        `fiftyone.zoo.download_zoo_dataset()`

```

**Examples**

```
# Download a zoo dataset
fiftyone zoo datasets download <name>

```

```
# Download a remotely-sourced zoo dataset
fiftyone zoo datasets download https://github.com/<user>/<repo>
fiftyone zoo datasets download <url>

```

```
# Download the specified split(s) of a zoo dataset
fiftyone zoo datasets download <name> --splits <split1> ...

```

```
# Download a zoo dataset that requires extra keyword arguments
fiftyone zoo datasets download <name> \
    --kwargs source_dir=/path/to/source/files

```

### Load zoo datasets [¶](\#load-zoo-datasets "Permalink to this headline")

Load zoo datasets as persistent FiftyOne datasets.

When loading remotely-sourced zoo datasets, you can provide any of the
following formats:

- a GitHub repo URL like `https://github.com/<user>/<repo>`

- a GitHub ref like `https://github.com/<user>/<repo>/tree/<branch>` or
`https://github.com/<user>/<repo>/commit/<commit>`

- a GitHub ref string like `<user>/<repo>[/<ref>]`

- a publicly accessible URL of an archive (eg zip or tar) file


Note

To download from a private GitHub repository that you have access to,
provide your GitHub personal access token by setting the `GITHUB_TOKEN`
environment variable.

```
fiftyone zoo datasets load [-h] [-s SPLITS [SPLITS ...]]
                           [-n DATASET_NAME] [-k KEY=VAL [KEY=VAL ...]]
                           NAME_OR_URL

```

**Arguments**

```
positional arguments:
  NAME_OR_URL           the name or remote location of the dataset

optional arguments:
  -h, --help            show this help message and exit
  -s SPLITS [SPLITS ...], --splits SPLITS [SPLITS ...]
                        the dataset splits to load
  -n DATASET_NAME, --dataset-name DATASET_NAME
                        a custom name to give the FiftyOne dataset
  -k KEY=VAL [KEY=VAL ...], --kwargs KEY=VAL [KEY=VAL ...]
                        additional dataset-specific keyword arguments for
                        `fiftyone.zoo.load_zoo_dataset()`

```

**Examples**

```
# Load the zoo dataset with the given name
fiftyone zoo datasets load <name>

```

```
# Load a remotely-sourced zoo dataset
fiftyone zoo datasets load https://github.com/<user>/<repo>
fiftyone zoo datasets load <url>

```

```
# Load the specified split(s) of a zoo dataset
fiftyone zoo datasets load <name> --splits <split1> ...

```

```
# Load a zoo dataset with a custom name
fiftyone zoo datasets load <name> --dataset-name <dataset-name>

```

```
# Load a zoo dataset that requires custom keyword arguments
fiftyone zoo datasets load <name> \
    --kwargs source_dir=/path/to/source_files

```

```
# Load a random subset of a zoo dataset
fiftyone zoo datasets load <name> \
    --kwargs max_samples=50 shuffle=True

```

### Delete zoo datasets [¶](\#delete-zoo-datasets "Permalink to this headline")

Deletes the local copy of the zoo dataset on disk.

```
fiftyone zoo datasets delete [-h] [-s SPLIT] NAME

```

**Arguments**

```
positional arguments:
  NAME                  the name of the dataset

optional arguments:
  -h, --help            show this help message and exit
  -s SPLIT, --split SPLIT
                        a dataset split

```

**Examples**

```
# Delete an entire zoo dataset from disk
fiftyone zoo datasets delete <name>

```

```
# Delete a specific split of a zoo dataset from disk
fiftyone zoo datasets delete <name> --split <split>

```

## FiftyOne Model Zoo [¶](\#fiftyone-model-zoo "Permalink to this headline")

Tools for working with the FiftyOne Model Zoo.

```
fiftyone zoo models [-h] [--all-help]
                    {list,find,info,requirements,download,apply,embed,delete,list-sources,register-source,delete-source}
                    ...

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  --all-help            show help recursively and exit

available commands:
  {list,find,info,requirements,download,apply,embed,delete,register-source,delete-source}
    list                List models in the FiftyOne Model Zoo.
    find                Locate the downloaded zoo model on disk.
    info                Print information about models in the FiftyOne Model Zoo.
    requirements        Handles package requirements for zoo models.
    download            Download zoo models.
    apply               Apply zoo models to datasets.
    embed               Generate embeddings for datasets with zoo models.
    delete              Deletes the local copy of the zoo model on disk.
    list-sources        Lists remote zoo model sources that are registered locally.
    register-source     Registers a remote source of zoo models.
    delete-source       Deletes the remote source and all downloaded models associated with it.

```

### List models in zoo [¶](\#list-models-in-zoo "Permalink to this headline")

List models in the FiftyOne Model Zoo.

```
fiftyone zoo models list [-h] [-n] [-d] [-t TAGS] [-s SOURCE]

```

**Arguments**

```
optional arguments:
  -h, --help            show this help message and exit
  -n, --names-only      only show model names
  -d, --downloaded-only
                        only show models that have been downloaded
  -t TAGS, --tags TAGS  only show models with the specified tag or list,of,tags
  -s SOURCE, --source SOURCE
                        only show models available from the specified remote source

```

**Examples**

```
# List available models
fiftyone zoo models list

```

```
# List available models (names only)
fiftyone zoo models list --names-only

```

```
# List downloaded models
fiftyone zoo models list --downloaded-only

```

```
# List available models with the given tag
fiftyone zoo models list --tags <tag>

```

```
# List available models from the given remote source
fiftyone zoo models list --source <source>

```

### Find zoo models on disk [¶](\#find-zoo-models-on-disk "Permalink to this headline")

Locate the downloaded zoo model on disk.

```
fiftyone zoo models find [-h] NAME

```

**Arguments**

```
positional arguments:
  NAME                  the name of the model

optional arguments:
  -h, --help            show this help message and exit

```

**Examples**

```
# Print the location of the downloaded zoo model on disk
fiftyone zoo models find <name>

```

### Show zoo model info [¶](\#show-zoo-model-info "Permalink to this headline")

Print information about models in the FiftyOne Model Zoo.

```
fiftyone zoo models info [-h] NAME

```

**Arguments**

```
positional arguments:
  NAME                  the name of the model

optional arguments:
  -h, --help            show this help message and exit

```

**Examples**

```
# Print information about a zoo model
fiftyone zoo models info <name>

```

### Zoo model requirements [¶](\#zoo-model-requirements "Permalink to this headline")

Handles package requirements for zoo models.

```
fiftyone zoo models requirements [-h] [-p] [-i] [-e]
                                 [--error-level LEVEL]
                                 NAME

```

**Arguments**

```
positional arguments:
  NAME                 the name of the model

optional arguments:
  -h, --help           show this help message and exit
  -p, --print          print the requirements for the zoo model
  -i, --install        install any requirements for the zoo model
  -e, --ensure         ensure the requirements for the zoo model are satisfied
  --error-level LEVEL  the error level (0=error, 1=warn, 2=ignore) to use
                       when installing or ensuring model requirements

```

**Examples**

```
# Print requirements for a zoo model
fiftyone zoo models requirements <name> --print

```

```
# Install any requirements for the zoo model
fiftyone zoo models requirements <name> --install

```

```
# Ensures that the requirements for the zoo model are satisfied
fiftyone zoo models requirements <name> --ensure

```

### Download zoo models [¶](\#download-zoo-models "Permalink to this headline")

Download zoo models.

When downloading remotely-sourced zoo models, you can provide any of the
following:

- a GitHub repo URL like `https://github.com/<user>/<repo>`

- a GitHub ref like `https://github.com/<user>/<repo>/tree/<branch>` or
`https://github.com/<user>/<repo>/commit/<commit>`

- a GitHub ref string like `<user>/<repo>[/<ref>]`


Note

To download from a private GitHub repository that you have access to,
provide your GitHub personal access token by setting the `GITHUB_TOKEN`
environment variable.

```
fiftyone zoo models download [-h] [-n MODEL_NAME] [-o] NAME_OR_URL

```

**Arguments**

```
positional arguments:
  NAME_OR_URL           the name or remote location of the model

optional arguments:
  -h, --help            show this help message and exit
  -n MODEL_NAME, --model-name MODEL_NAME
                        the specific model to download, if `name_or_url` is
                        a remote source
  -o, --overwrite       whether to overwrite any existing model files

```

**Examples**

```
# Download a zoo model
fiftyone zoo models download <name>

```

```
# Download a remotely-sourced zoo model
fiftyone zoo models download https://github.com/<user>/<repo> \
    --model-name <name>
fiftyone zoo models download <url> --model-name <name>

```

### Apply zoo models to datasets [¶](\#apply-zoo-models-to-datasets "Permalink to this headline")

Apply zoo models to datasets.

When applying remotely-sourced zoo models, you can provide any of the following
formats:

- a GitHub repo URL like `https://github.com/<user>/<repo>`

- a GitHub ref like `https://github.com/<user>/<repo>/tree/<branch>` or
`https://github.com/<user>/<repo>/commit/<commit>`

- a GitHub ref string like `<user>/<repo>[/<ref>]`

- a publicly accessible URL of an archive (eg zip or tar) file


Note

To download from a private GitHub repository that you have access to,
provide your GitHub personal access token by setting the `GITHUB_TOKEN`
environment variable.

```
fiftyone zoo models apply [-h] [-n MODEL_NAME] [-b BATCH_SIZE] [-t THRESH]
                          [-l] [-i] [--error-level LEVEL]
                          NAME_OR_URL DATASET_NAME LABEL_FIELD

```

**Arguments**

```
positional arguments:
  NAME_OR_URL           the name or remote location of the zoo model
  DATASET_NAME          the name of the FiftyOne dataset to process
  LABEL_FIELD           the name of the field in which to store the predictions

optional arguments:
  -h, --help            show this help message and exit
  -n MODEL_NAME, --model-name MODEL_NAME
                        the specific model to apply, if `name_or_url` is a
                        remote source
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        an optional batch size to use during inference
  -t THRESH, --confidence-thresh THRESH
                        an optional confidence threshold to apply to any
                        applicable labels generated by the model
  -l, --store-logits    store logits for the predictions
  -i, --install         install any requirements for the zoo model
  --error-level LEVEL   the error level (0=error, 1=warn, 2=ignore) to use
                        when installing or ensuring model requirements

```

**Examples**

```
# Apply a zoo model to a dataset
fiftyone zoo models apply <model-name> <dataset-name> <label-field>

```

```
# Apply a remotely-sourced zoo model to a dataset
fiftyone zoo models apply https://github.com/<user>/<repo> \
    <dataset-name> <label-field> --model-name <model-name>
fiftyone zoo models apply <url> \
    <dataset-name> <label-field> --model-name <model-name>

```

```
# Apply a zoo classifier with some customized parameters
fiftyone zoo models apply \
    <model-name> <dataset-name> <label-field> \
    --confidence-thresh 0.7 \
    --store-logits \
    --batch-size 32

```

### Generate embeddings with zoo models [¶](\#generate-embeddings-with-zoo-models "Permalink to this headline")

Generate embeddings for datasets with zoo models.

When applying remotely-sourced zoo models, you can provide any of the following
formats:

- a GitHub repo URL like `https://github.com/<user>/<repo>`

- a GitHub ref like `https://github.com/<user>/<repo>/tree/<branch>` or
`https://github.com/<user>/<repo>/commit/<commit>`

- a GitHub ref string like `<user>/<repo>[/<ref>]`

- a publicly accessible URL of an archive (eg zip or tar) file


Note

To download from a private GitHub repository that you have access to,
provide your GitHub personal access token by setting the `GITHUB_TOKEN`
environment variable.

```
fiftyone zoo models embed [-h] [-n MODEL_NAME] [-b BATCH_SIZE] [-i]
                          [--error-level LEVEL]
                          NAME_OR_URL DATASET_NAME EMBEDDINGS_FIELD

```

**Arguments**

```
positional arguments:
  NAME_OR_URL           the name or remote location of the zoo model
  DATASET_NAME          the name of the FiftyOne dataset to process
  EMBEDDINGS_FIELD      the name of the field in which to store the embeddings

optional arguments:
  -h, --help            show this help message and exit
  -n MODEL_NAME, --model-name MODEL_NAME
                        the specific model to apply, if `name_or_url` is a
                        remote source
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        an optional batch size to use during inference
  -i, --install         install any requirements for the zoo model
  --error-level LEVEL   the error level (0=error, 1=warn, 2=ignore) to use
                        when installing or ensuring model requirements

```

**Examples**

```
# Generate embeddings for a dataset with a zoo model
fiftyone zoo models embed <model-name> <dataset-name> <embeddings-field>

```

```
# Generate embeddings for a dataset with a remotely-sourced zoo model
fiftyone zoo models embed https://github.com/<user>/<repo> \
    <dataset-name> <embeddings-field> --model-name <model-name>
fiftyone zoo models embed <url> \
    <dataset-name> <embeddings-field> --model-name <model-name>

```

### Delete zoo models [¶](\#delete-zoo-models "Permalink to this headline")

Deletes the local copy of the zoo model on disk.

```
fiftyone zoo models delete [-h] NAME

```

**Arguments**

```
positional arguments:
  NAME        the name of the model

optional arguments:
  -h, --help  show this help message and exit

```

**Examples**

```
# Delete the zoo model from disk
fiftyone zoo models delete <name>

```

### List zoo model sources [¶](\#list-zoo-model-sources "Permalink to this headline")

Lists remote zoo model sources that are registered locally.

```
fiftyone zoo models list-sources [-h]

```

**Examples**

```
# Lists the registered remote zoo model sources
fiftyone zoo models list-sources

```

### Register zoo model sources [¶](\#register-zoo-model-sources "Permalink to this headline")

Registers a remote source of zoo models.

You can provide any of the following formats:

- a GitHub repo URL like `https://github.com/<user>/<repo>`

- a GitHub ref like `https://github.com/<user>/<repo>/tree/<branch>` or
`https://github.com/<user>/<repo>/commit/<commit>`

- a GitHub ref string like `<user>/<repo>[/<ref>]`

- a publicly accessible URL of an archive (eg zip or tar) file


Note

To download from a private GitHub repository that you have access to,
provide your GitHub personal access token by setting the `GITHUB_TOKEN`
environment variable.

```
fiftyone zoo models register-source [-h] [-o] URL_OR_GH_REPO

```

**Arguments**

```
positional arguments:
  URL_OR_GH_REPO   the remote source to register

optional arguments:
  -h, --help       show this help message and exit
  -o, --overwrite  whether to overwrite any existing files

```

**Examples**

```
# Register a remote zoo model source
fiftyone zoo models register-source https://github.com/<user>/<repo>
fiftyone zoo models register-source <url>

```

### Delete zoo model sources [¶](\#delete-zoo-model-sources "Permalink to this headline")

Deletes the remote source and all downloaded models associated with it.

You can provide any of the following formats:

- a GitHub repo URL like `https://github.com/<user>/<repo>`

- a GitHub ref like `https://github.com/<user>/<repo>/tree/<branch>` or
`https://github.com/<user>/<repo>/commit/<commit>`

- a GitHub ref string like `<user>/<repo>[/<ref>]`

- a publicly accessible URL of an archive (eg zip or tar) file


```
fiftyone zoo models delete-source [-h] URL_OR_GH_REPO

```

**Arguments**

```
positional arguments:
  URL_OR_GH_REPO   the remote source to delete

optional arguments:
  -h, --help       show this help message and exit

```

**Examples**

```
# Delete a remote zoo model source
fiftyone zoo models delete-source https://github.com/<user>/<repo>
fiftyone zoo models delete-source <url>

```

- FiftyOne Command-Line Interface (CLI)
  - [Quickstart](#quickstart)
    - [Tab completion](#tab-completion)
  - [FiftyOne CLI](#cli-fiftyone-main)
  - [FiftyOne quickstart](#fiftyone-quickstart)
  - [FiftyOne config](#fiftyone-config)
  - [Print constants](#print-constants)
  - [Convert dataset formats](#convert-dataset-formats)
  - [FiftyOne datasets](#fiftyone-datasets)
    - [List datasets](#list-datasets)
    - [Print dataset information](#print-dataset-information)
    - [Print dataset stats](#print-dataset-stats)
    - [Create datasets](#create-datasets)
    - [Print dataset head](#print-dataset-head)
    - [Print dataset tail](#print-dataset-tail)
    - [Stream samples to the terminal](#stream-samples-to-the-terminal)
    - [Export datasets](#export-datasets)
    - [Drawing labels on samples](#drawing-labels-on-samples)
    - [Rename datasets](#rename-datasets)
    - [Delete datasets](#delete-datasets)
  - [FiftyOne migrations](#fiftyone-migrations)
  - [FiftyOne operators](#fiftyone-operators)
    - [List operators](#list-operators)
    - [Operator info](#operator-info)
  - [FiftyOne delegated operations](#fiftyone-delegated-operations)
    - [Launch delegated service](#launch-delegated-service)
    - [List delegated operations](#list-delegated-operations)
    - [Delegated operation info](#delegated-operation-info)
    - [Mark delegated operations as failed](#mark-delegated-operations-as-failed)
    - [Delete delegated operations](#delete-delegated-operations)
    - [Cleanup delegated operations](#cleanup-delegated-operations)
  - [FiftyOne plugins](#fiftyone-plugins)
    - [List plugins](#list-plugins)
    - [Plugin info](#plugin-info)
    - [Download plugins](#download-plugins)
    - [Plugin requirements](#plugin-requirements)
    - [Create plugins](#create-plugins)
    - [Enable plugins](#enable-plugins)
    - [Disable plugins](#disable-plugins)
    - [Delete plugins](#delete-plugins)
  - [FiftyOne utilities](#fiftyone-utilities)
    - [Compute metadata](#compute-metadata)
    - [Transform images](#transform-images)
    - [Transform videos](#transform-videos)
  - [FiftyOne Annotation](#fiftyone-annotation)
    - [Annotation Config](#annotation-config)
  - [FiftyOne App](#fiftyone-app)
    - [App Config](#app-config)
    - [Launch the App](#launch-the-app)
    - [View datasets in App](#view-datasets-in-app)
    - [Connect to remote App](#connect-to-remote-app)
  - [FiftyOne Brain](#fiftyone-brain)
    - [Brain Config](#brain-config)
  - [FiftyOne Evaluation](#fiftyone-evaluation)
    - [Evaluation Config](#evaluation-config)
  - [FiftyOne Zoo](#fiftyone-zoo)
  - [FiftyOne Dataset Zoo](#fiftyone-dataset-zoo)
    - [List datasets in zoo](#list-datasets-in-zoo)
    - [Find zoo datasets on disk](#find-zoo-datasets-on-disk)
    - [Show zoo dataset info](#show-zoo-dataset-info)
    - [Download zoo datasets](#download-zoo-datasets)
    - [Load zoo datasets](#load-zoo-datasets)
    - [Delete zoo datasets](#delete-zoo-datasets)
  - [FiftyOne Model Zoo](#fiftyone-model-zoo)
    - [List models in zoo](#list-models-in-zoo)
    - [Find zoo models on disk](#find-zoo-models-on-disk)
    - [Show zoo model info](#show-zoo-model-info)
    - [Zoo model requirements](#zoo-model-requirements)
    - [Download zoo models](#download-zoo-models)
    - [Apply zoo models to datasets](#apply-zoo-models-to-datasets)
    - [Generate embeddings with zoo models](#generate-embeddings-with-zoo-models)
    - [Delete zoo models](#delete-zoo-models)
    - [List zoo model sources](#list-zoo-model-sources)
    - [Register zoo model sources](#register-zoo-model-sources)
    - [Delete zoo model sources](#delete-zoo-model-sources)