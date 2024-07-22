# Finetuning starter kit

This code is built on the HuggingFace trainers and allows you to do any combination of:
- training/finetuning any causal LM from the HF Model Hub on the HackerCup dataset
- evaluating on the same
- sampling n code generations from the model checkpoint
- running all generations on the sample input/output pairs, and choosing the highest scoring one to run the full input/output pairs

## Installation

Install requirements with
```
pip install -r requirements.txt
```
An environment manager (e.g. conda) is recommended. Depending on your exact CUDA version and other factors, you may need to install some requirements manually.

## Training and evaluation

### To evaluate a pretrained model on HackerCup by generating n solutions and selecting the best

```
python train_and_eval.py --output_dir ~/temp --num_gens 10 --max_code_gen_examples -1
```
`--output_dir` is required but is not used if not finetuning.

We currently filter to only include problems from HackerCup for which each test case occupies exactly 
one input and one output line, as we are only providing a starting point. This significantly reduces 
the number of available examples, from 284 to 17. Generally, input parsing and formatting 
should be handled by the model, unlike in this starter kit example, which serves only as a starting point.

Command-line arguments controlling some data and model hyperparameters are defined at the top of `train_and_eval.py`. 
The most relevant may include:
- `--text_columns`, the set of columns included in the input to the model. 
- `--num_gens`, the number of generations to sample for each problem.
- `--max_code_gen_examples`, the number of HackerCup problems to sample code generations for.
- `--model_name_or_path`, the HuggingFace model to use. Some models have context length constraints, which will result in errors unless you modify this code to shorten the inputs.

### To finetune a pretrained model on HackerCup dataset and evaluate as above
```
python train_and_eval.py --do_train --do_eval --output_dir <OUTPUT_DIR> --num_gens 10
```
This command treats the HackerCup dataset as a language modeling task, by which we mean that loss is applied to all tokens. It is possible to only apply loss to tokens in the code solutions by using a [`trl.SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTTrainer). A sketch of this approach is included in `train_hf_model()` and is triggered with `--train_task seq2seq`, but is not complete. 

Command-line arguments controlling some data and model hyperparameters are defined at the top of `train_and_eval.py`. 
We also use the HuggingFace `transformer.TrainingArguments` hyperparameters. A [list of options](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) including train epochs, batch size, and 
learning rate can be found in the transformers documentation.

### To adapt this code for finetuning on other HF datasets

The dataset to train on is determined by `--dataset_name`.

Modifications need to be made to `format_datasets()`, which is currently written 
specifically to ingest examples in the HackerCup format. For some datasets, this can 
simply return the `text` field of an example.

Some command line arguments, such as `--text_columns` need to be adjusted for the columns in your dataset.