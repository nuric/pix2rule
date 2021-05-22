# pix2rule

This is the source code repository that accompanies paper titled "pix2rule: End-to-end Neuro-symbolic Rule Learning". It contains the models, dataset processing pipelines as well as the trianing scripts. It also contains experimental code that was used during the research process but are not integrated into the final results.

## Project Structure

The project contains some reusable parts and some that are integrated into the local repository structure. Mainly it revolves around the following folders and files:

- **components:** Contains reusable layers and operations. Every component is unit tested and self-isolated. It contains things layers such as CNN backbones and object selection. Note that not all components are used in the final models.
- **datasets:** Describes the available datasets, how to generate and load them. Every dataset gets its own module and the `__init__.py` keeps track of them.
- **models:** Similar to datasets, it contains models that often use the available components. They are full trainable models and use the `build_model` function to create them. The module `__init__.py` file keeps track of available models.
- **utils:** Contains utility functions and classes such as custom callbacks. Some are tested such as dictionary hashing. Utility functions are also self-contained and can be used in other projects.
- **configlib:** The mini configuration library that handles multiple file `argparse` flags. It collects all the arguments, parses them and provides extra functionality such as reading in a JSON file. Utilities and components do not use this library to make them self-contained.
- **reportlib:** The mini tensor and model reporting library for extracting out intermediate tensors and representations. It provides two main parts: `report_tensor` and `ReportLayer` which store passed tensor values to a global dictionary when run with `create_report`. Note this requires the model to be run in eager mode so the intermediate tensors are evaluated.
- **train** The entry point for running experiments. One can find out all moving parts by running `python3 train.py -h`. It will list all collected flags. It loads the dataset, create the model, sets up mlflow and trains the model.
- **experiments:** A script to generate experiment configurations and pre-generated required datasets. It will list all experiment configurations and write them into a file to run later.
- **condor_run:** Takes the experiment configurations generated by experiments and runs them using the department HTCondor cluster.
- **tasks:** Build tools and cleaning tasks using pyinvoke.

## Dependencies, unit testing, linting, type checking

The dependencies for the respository can be installed using:

```bash
pip3 install -r requirements.txt
```

and the `ci_requirements.txt` are a minimal set of dependencies to run the continous integration pipeline below.

There are style checks, linting, type checking and unit tests maintained to ensure the integrity of this repository. The entire pipeline can be run using:

```bash
invoke build
```

after installing `pyinvoke` dependency.

## DNF Layer

The main contribution of the paper is the DNF Layer located in `components/dnf_layer.py` and in particular the `WeightedDNF` layer. It stacks two semi-symbolic layers and handles the permutation as well as the existential reduction. You'll also notice other implementations that we tried but decided not to purpsue in the final results. Along with the BaseDNF, you can reuse this layer as a TensorFlow Keras layer.

## Datasets

There are dataset used: subgraph set isomorphism which essentially generates disjunctive normal form formulas based on graph theory, and relations game image classification dataset. These are contained in `datasets/gendnf.py` and `datasets/relsgame.py` which generate, save and load the data required. They are coordinated by the training script, described below.

## Experiments

All experiment configurations are generated using the `experiments.py` file. It will configure every run that needs to happen and makes use of the `utils` module such as dictionary hashing and hyper-parameter generation tool `utils/hyperrun.py`. When the script is run, it will generate an `experiments.json` file which contains all configurations that need to be run.

## Training

The main entry point is the `train.py` script which curates all the arguments, loads the dataset, loads the model and runs the training and evaluation stages. Here are all the arguments that can be passed to it:

```
>>> python3 train.py -h

usage: experiments.py [-h] [--config_json CONFIG_JSON] Configuration for experiments.

optional arguments:
  -h, --help            show this help message and exit
  --config_json CONFIG_JSON
                        Configuration json and index to merge. (default: None)

Relsgame Dataset config:

  --relsgame_tasks [{same,between,occurs,xoccurs} [{same,between,occurs,xoccurs} ...]]
                        Task names to generate, empty list for all. (default: [])
  --relsgame_train_size RELSGAME_TRAIN_SIZE
                        Relations game training size per task, 0 to use everything. (default: 1000)
  --relsgame_validation_size RELSGAME_VALIDATION_SIZE
                        Validation size per task, 0 to use everything. (default: 1000)
  --relsgame_test_size RELSGAME_TEST_SIZE
                        Test size per task, 0 to use everything. (default: 1000)
  --relsgame_batch_size RELSGAME_BATCH_SIZE
                        Data batch size. (default: 64)
  --relsgame_output_type {label,image,onehot_label,label_and_image,onehot_label_and_image}
                        Type of prediction task. (default: label)
  --relsgame_with_augmentation
                        Apply data augmentation. (default: False)
  --relsgame_noise_stddev RELSGAME_NOISE_STDDEV
                        Added noise to image at training inputs. (default: 0.0)
  --relsgame_rng_seed RELSGAME_RNG_SEED
                        Random number generator seed for data augmentation. (default: 42)

DNF dataset config:

  --gendnf_num_objects GENDNF_NUM_OBJECTS
                        Number of objects / constants. (default: 4)
  --gendnf_num_nullary GENDNF_NUM_NULLARY
                        Number of nullary predicates. (default: 6)
  --gendnf_num_unary GENDNF_NUM_UNARY
                        Number of unary predicates in the language. (default: 7)
  --gendnf_num_binary GENDNF_NUM_BINARY
                        Number of binary predicates in the language. (default: 8)
  --gendnf_num_variables GENDNF_NUM_VARIABLES
                        Number of variables in rule body. (default: 3)
  --gendnf_num_conjuncts GENDNF_NUM_CONJUNCTS
                        Number of ways a rule can be defined. (default: 4)
  --gendnf_target_arity GENDNF_TARGET_ARITY
                        Arity of target rule to learn. (default: 0)
  --gendnf_gen_size GENDNF_GEN_SIZE
                        Number of examples per label to generate. (default: 10000)
  --gendnf_train_size GENDNF_TRAIN_SIZE
                        DNF training size, number of positive + negative examples upper bound. (default: 1000)
  --gendnf_validation_size GENDNF_VALIDATION_SIZE
                        Validation size positive + negative examples upper bound. (default: 1000)
  --gendnf_test_size GENDNF_TEST_SIZE
                        Test size upper bound. (default: 1000)
  --gendnf_batch_size GENDNF_BATCH_SIZE
                        Data batch size. (default: 64)
  --gendnf_input_noise_probability GENDNF_INPUT_NOISE_PROBABILITY
                        Added input noise probability for training set. (default: 0.0)
  --gendnf_rng_seed GENDNF_RNG_SEED
                        Random number generator seed. (default: 42)
  --gendnf_return_numpy
                        Return raw numpy arrays instead of tf data. (default: False)

Data config:

  --dataset_name {relsgame,gendnf}
                        Dataset name to train / evaluate. (default: relsgame)

DNF Image Model Options:

  --dnf_image_classifier_image_layer_name {RelationsGameImageInput,RelationsGamePixelImageInput}
                        Image input layer to use. (default: RelationsGameImageInput)
  --dnf_image_classifier_image_hidden_size DNF_IMAGE_CLASSIFIER_IMAGE_HIDDEN_SIZE
                        Hidden size of image pipeline layers. (default: 32)
  --dnf_image_classifier_image_activation {relu,sigmoid,tanh}
                        Activation of hidden and final layer of image pipeline. (default: relu)
  --dnf_image_classifier_image_with_position
                        Append position coordinates. (default: False)
  --dnf_image_classifier_object_sel_layer_name {RelaxedObjectSelection,TopKObjectSelection}
                        Selection layer to use. (default: RelaxedObjectSelection)
  --dnf_image_classifier_object_sel_num_select DNF_IMAGE_CLASSIFIER_OBJECT_SEL_NUM_SELECT
                        Number of object to select. (default: 2)
  --dnf_image_classifier_object_sel_initial_temperature DNF_IMAGE_CLASSIFIER_OBJECT_SEL_INITIAL_TEMPERATURE
                        Initial selection temperature if layer uses it. (default: 0.5)
  --dnf_image_classifier_object_feat_layer_name {LinearObjectFeatures}
                        Selection layer to use. (default: LinearObjectFeatures)
  --dnf_image_classifier_object_feat_unary_size DNF_IMAGE_CLASSIFIER_OBJECT_FEAT_UNARY_SIZE
                        Number of unary predicates for objects. (default: 4)
  --dnf_image_classifier_object_feat_binary_size DNF_IMAGE_CLASSIFIER_OBJECT_FEAT_BINARY_SIZE
                        Number of binary predicates for objects. (default: 8)
  --dnf_image_classifier_object_feat_activation DNF_IMAGE_CLASSIFIER_OBJECT_FEAT_ACTIVATION
                        Activation of learnt predicates. (default: sigmoid)
  --dnf_image_classifier_hidden_layer_name {DNF,RealDNF,WeightedDNF}
                        DNF layer to use. (default: WeightedDNF)
  --dnf_image_classifier_hidden_arities [DNF_IMAGE_CLASSIFIER_HIDDEN_ARITIES [DNF_IMAGE_CLASSIFIER_HIDDEN_ARITIES ...]]
                        Number of predicates and their arities. (default: [0])
  --dnf_image_classifier_hidden_num_total_variables DNF_IMAGE_CLASSIFIER_HIDDEN_NUM_TOTAL_VARIABLES
                        Number of variables in conjunctions. (default: 2)
  --dnf_image_classifier_hidden_num_conjuncts DNF_IMAGE_CLASSIFIER_HIDDEN_NUM_CONJUNCTS
                        Number of conjunctions in this DNF. (default: 8)
  --dnf_image_classifier_hidden_recursive DNF_IMAGE_CLASSIFIER_HIDDEN_RECURSIVE
                        Whether the inputs contain layer outputs. (default: False)
  --dnf_image_classifier_inference_layer_name {DNF,RealDNF,WeightedDNF}
                        DNF layer to use. (default: WeightedDNF)
  --dnf_image_classifier_inference_arities [DNF_IMAGE_CLASSIFIER_INFERENCE_ARITIES [DNF_IMAGE_CLASSIFIER_INFERENCE_ARITIES ...]]
                        Number of predicates and their arities. (default: [0])
  --dnf_image_classifier_inference_num_total_variables DNF_IMAGE_CLASSIFIER_INFERENCE_NUM_TOTAL_VARIABLES
                        Number of variables in conjunctions. (default: 2)
  --dnf_image_classifier_inference_num_conjuncts DNF_IMAGE_CLASSIFIER_INFERENCE_NUM_CONJUNCTS
                        Number of conjunctions in this DNF. (default: 8)
  --dnf_image_classifier_inference_recursive DNF_IMAGE_CLASSIFIER_INFERENCE_RECURSIVE
                        Whether the inputs contain layer outputs. (default: False)
  --dnf_image_classifier_iterations DNF_IMAGE_CLASSIFIER_ITERATIONS
                        Number of inference steps to perform. (default: 1)

DNF Rule Model Options:

  --dnf_rule_learner_inference_layer_name {DNF,RealDNF,WeightedDNF}
                        DNF layer to use. (default: WeightedDNF)
  --dnf_rule_learner_inference_arities [DNF_RULE_LEARNER_INFERENCE_ARITIES [DNF_RULE_LEARNER_INFERENCE_ARITIES ...]]
                        Number of predicates and their arities. (default: [0])
  --dnf_rule_learner_inference_num_total_variables DNF_RULE_LEARNER_INFERENCE_NUM_TOTAL_VARIABLES
                        Number of variables in conjunctions. (default: 2)
  --dnf_rule_learner_inference_num_conjuncts DNF_RULE_LEARNER_INFERENCE_NUM_CONJUNCTS
                        Number of conjunctions in this DNF. (default: 8)
  --dnf_rule_learner_inference_recursive DNF_RULE_LEARNER_INFERENCE_RECURSIVE
                        Whether the inputs contain layer outputs. (default: False)

Predinet Image Model Options.:

  --predinet_image_layer_name {RelationsGameImageInput,RelationsGamePixelImageInput}
                        Image input layer to use. (default: RelationsGameImageInput)
  --predinet_image_hidden_size PREDINET_IMAGE_HIDDEN_SIZE
                        Hidden size of image pipeline layers. (default: 32)
  --predinet_image_activation {relu,sigmoid,tanh}
                        Activation of hidden and final layer of image pipeline. (default: relu)
  --predinet_image_with_position
                        Append position coordinates. (default: False)
  --predinet_relations PREDINET_RELATIONS
                        Number of relations to compute between features. (default: 4)
  --predinet_heads PREDINET_HEADS
                        Number of relation heads. (default: 4)
  --predinet_key_size PREDINET_KEY_SIZE
                        Number of relation heads. (default: 4)
  --predinet_output_hidden_size PREDINET_OUTPUT_HIDDEN_SIZE
                        MLP hidden layer size. (default: 8)

MLP Image Model Options.:

  --mlp_image_classifier_image_layer_name {RelationsGameImageInput,RelationsGamePixelImageInput}
                        Image input layer to use. (default: RelationsGameImageInput)
  --mlp_image_classifier_image_hidden_size MLP_IMAGE_CLASSIFIER_IMAGE_HIDDEN_SIZE
                        Hidden size of image pipeline layers. (default: 32)
  --mlp_image_classifier_image_activation {relu,sigmoid,tanh}
                        Activation of hidden and final layer of image pipeline. (default: relu)
  --mlp_image_classifier_image_with_position
                        Append position coordinates. (default: False)
  --mlp_image_classifier_hidden_sizes MLP_IMAGE_CLASSIFIER_HIDDEN_SIZES [MLP_IMAGE_CLASSIFIER_HIDDEN_SIZES ...]
                        Hidden layer sizes, length determines number of layers. (default: [32])
  --mlp_image_classifier_hidden_activations MLP_IMAGE_CLASSIFIER_HIDDEN_ACTIVATIONS [MLP_IMAGE_CLASSIFIER_HIDDEN_ACTIVATIONS ...]
                        Hidden layer activations, must match hidden_sizes. (default: ['relu'])

Slot attention auto encoder parameters:

  --slotae_image_layer_name {RelationsGameImageInput,RelationsGamePixelImageInput}
                        Image input layer to use. (default: RelationsGameImageInput)
  --slotae_image_hidden_size SLOTAE_IMAGE_HIDDEN_SIZE
                        Hidden size of image pipeline layers. (default: 32)
  --slotae_image_activation {relu,sigmoid,tanh}
                        Activation of hidden and final layer of image pipeline. (default: relu)
  --slotae_image_with_position
                        Append position coordinates. (default: False)

Global model options:

  --model_name {dnf_image_classifier,dnf_rule_learner,predinet,slot_ae,mlp_image_classifier}
                        Model name to train / evaluate. (default: dnf_image_classifier)

Pix2Rule options:

  --experiment_name EXPERIMENT_NAME
```

Note that not all arguments are used. For example, if the dataset is set to relations game, the arguments for the generated DNF data of the subgraph set isomorphism will be ignored.

Once the experiments are generated they can be run using the `--config_json` argument, for example:

```bash
python3 train.py --config_json data/experiments.json.0abae776c4ceaf08bd870f41d7236c59
``` 

where the last string is the configuration hash that uniquely identifies the experiment to run from the `data/experiments.json` file.

## Built With

  - [TensorFlow](https://tensorflow.org) - deep learning framework
  - [mlflow](https://mlflow.org) - experiment tracking platform
  - [Matplotlib](https://matplotlib.org/) - main plotting library
  - [seaborn](https://seaborn.pydata.org/) - helper plotting library for some charts
  - [NumPy](http://www.numpy.org/) - main numerical library for data vectorisation
  - [Pandas](https://pandas.pydata.org/) - helper data manipulation library
  - [jupyter](https://jupyter.org) - interactive environment to analyse data / results
  - [black](https://github.com/psf/black) - code styling tool used
  - [pylint](https://pylint.org) - linting tool for code integrity
  - [mypy](https://mypy.readthedocs.io/en/stable/) - type checking tool
  - [pyinvoke](http://www.pyinvoke.org/) task automation library used for the build process
  - [tqdm](https://tqdm.github.io/) progress bar library
