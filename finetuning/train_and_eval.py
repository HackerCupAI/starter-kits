import math
import os
import subprocess
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional

import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

os.environ["TOKENIZERS_PARALLELISM"] = "false" # required to use subprocess

MAX_TIME = 600              # Maximum time in seconds to generate a program
MAX_NEW_TOKENS = 500        # Maximum number of tokens to generate
DEFAULT_BLOCK_SIZE = 1024   # Default block size for LM training
MAX_CONTEXT_LENGTH = 16384  # Maximum number of tokens in the context
DATASET_PATH = "/fsx-onellm/margaretli/code/hackercup" # Path to the HackerCup dataset raw files

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="codellama/CodeLlama-7b-Python-hf",
        # default="gpt2", # used for debugging
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="hackercupai/hackercup", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_cache_dir: Optional[str] = field(
        default="datasets",
        metadata={"help": "The directory in which to cache the dataset and tokenizer."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=1000,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=1000,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=DEFAULT_BLOCK_SIZE,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    validation_split_percentage: Optional[int] = field(
        default=20,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    train_obj: str = field(
        default="lm", metadata={"help": "The objective to train the model on. Only lm and seq2seq are supported."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    text_cols: str = field(
        default="statement,sample_input,sample_output,code",
        # default="sample_input,sample_output,code",
        metadata={
            "help": (
                "Comma-separated string of columns in the dataset to use for text generation tasks. "
                "The hackercup dataset includes the columns: "
                "name, statement, input, solution, code, output, sample_input, sample_output, images"
                "Only name, statement, code, sample_input, and sample_output are currently supported. "
                "Modify this by updating the format_example() function."
            )
        },
    )

    def __post_init__(self):

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
        if self.text_cols is not None:
            self.text_cols = self.text_cols.split(",")
        assert all([
            col in [
                "name", "statement", "input", "solution", "code", "output", "sample_input", "sample_output", "images"
            ] for col in self.text_cols]), "Invalid column name in `text_cols`, must be supported by HackerCup dataset"
        assert self.train_obj in ["lm", "seq2seq"], "`train_obj` should be either 'lm' or 'seq2seq'."

@dataclass
class ProgramGenArguments:
    max_time: Optional[int] = field(
        default=MAX_TIME,
        metadata={"help": "The maximum time in seconds to generate a program."},
    )
    max_new_tokens: Optional[int] = field(
        default=MAX_NEW_TOKENS,
        metadata={"help": "The maximum number of new tokens to generate."},
    )
    num_gens: Optional[int] = field(
        default=1,
        metadata={"help": "The number of programs to generate per HackerCup problem."},
    )
    max_code_gen_examples: Optional[int] = field(
        default=-1,
        metadata={"help": (
            "The maximum number of HackerCup problems to generate solutions for. "
            "Default is set to -1 to generate for all problems."
            )
        },
    )
    run_generated_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the generated code on the sample input."},
    )
    only_run_generated_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to only run previously generated code on the sample input."},
    )
    gen_lang: Optional[str] = field(
        default="Python",
        metadata={"help": "The language to generate programs in."},
    )
    programs_path: Optional[str] = field(
        default="programs",
        metadata={"help": "The directory to save the generated programs."},
    )

def filter_examples(datasets, require_text_cols=None, require_input_output=True, only_use_single_line_samples=True):
    if require_text_cols:
        datasets = datasets.filter(lambda example: all([example[col] for col in require_text_cols]))
    if require_input_output:
        datasets = datasets.filter(lambda example: all([example[col] for col in ["input", "output"]]))
    if only_use_single_line_samples:
        datasets = datasets.filter(lambda example: int(example['input'].split('\n')[0]) == len(example['output'].strip().split('\n')))
        datasets = datasets.filter(lambda example: int(example['input'].split('\n')[0]) == len(example['input'].strip().split('\n')) - 1)
    return datasets


def format_example(examples, text_cols, max_sample_pairs=5, max_sample_length=100, include_output=True, language=""):
    inputs = []
    ids = []
    for i in range(len(examples['name'])):
        input = "### Problem:\n"
        if 'name' in text_cols and examples['name'][i]:
            input += f"## Name: {examples['name'][i]}\n"
        if 'statement' in text_cols and examples['statement'][i]:
            statement_str = examples['statement'][i].replace('\n', '\n## ')
            input += f"## Statement: {statement_str}\n"
        if "sample_input" in text_cols and "sample_output" in text_cols and examples['sample_input'][i] and examples['sample_output'][i]:
            num_samples = 0
            input += "### Examples:\n"
            for inp, outp in zip(examples['sample_input'][i].split('\n')[1:], examples['sample_output'][i].split('\n')):
                if not inp and not outp:
                    continue
                output_str = outp.split(":", 1)[1].strip() # remove the "Case #n: " prefix
                sample_str = f"##    >>> f({inp})\n##    {output_str}\n"
                if len(sample_str) > max_sample_length:
                    continue
                input += sample_str
                num_samples += 1
                if num_samples == max_sample_pairs:
                    break
        input += f"## Write a {language + ' ' if language else ''}function that takes a single argument and returns the correct output for the examples given.\n"
        input += f"### {language + ' ' if language else ''}Code: \n"
        if include_output:
            input += examples['code'][i]
        inputs.append(input)
        ids.append(f"{examples['year'][i]}/{examples['round'][i]}/{examples['name'][i]}")
    return inputs, ids


def write_programs(
        prompts, names, pipeline, tokenizer,
        max_new_tokens, max_time, num_gens_per_prompt=1, 
        language="Python", programs_path="programs", max_code_gen_examples=-1, 
    ):
    python_prefix = "def f(a):\n    "
    prefix = ""
    for i, (name, prompt) in enumerate(zip(names, prompts)):
        if max_code_gen_examples > 0 and i >= max_code_gen_examples:
            break
        if language == "Python":
            prefix = python_prefix
        prompt = f"{prompt}\n{prefix}" if prefix else prompt
        seqs = pipeline(
            prompt,
            do_sample=True,
            # temperature=0.1,
            top_p=0.9,
            num_return_sequences=num_gens_per_prompt,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
            # stop_strings=["def "],  # stop if second func started
            max_time=max_time,  # seconds
            return_full_text=False,
        )
        full_results = [seq["generated_text"] for seq in seqs]

        for num, full_result in enumerate(full_results):
            result = prefix + full_result.split("def ", 1)[0] # only take the first function -- hacky
            p_program = f"{programs_path}/{name}/{str(num)}.py"
            p_program = Path(p_program)
            p_program.parent.mkdir(parents=True, exist_ok=True)
            p_program.write_text(result + template)


template = """
T = int(input())
for case_num in range(1, T + 1):
    a = input().split()
    for i in range(len(a)):
        try:
            a[i] = float(a[i])
        except ValueError:
            pass
        try:
            a[i] = int(a[i])
        except ValueError:
            pass
    if len(a) == 1:
        a = a[0]
    else:
        # Simplify when first item is number of strings in list
        if isinstance(a[0], int) and isinstance(a[1], str):
            if (len(a) - 1) == a[0]:
                a = a[1:]
    print(f"Case #{case_num}: {f(a)}")
"""

def evaluate_programs(
        names, programs_path="programs", dataset_path=DATASET_PATH, language="Python"
    ):
    results = {}
    
    for name in names:
        best_program = None
        best_score = -1
        p_in = f"{dataset_path}/{name}.in"
        p_out = f"{dataset_path}/{name}.out"
        p_sample_in = f"{dataset_path}/{name}_sample_input.txt"
        p_sample_out = f"{dataset_path}/{name}_sample_output.txt"
        with open(p_out, "r") as f_out:
            out = f_out.readlines()
        with open(p_sample_out, "r") as f_sample_out:
            sample_out = f_sample_out.readlines()
        for p_program in Path(programs_path).glob(f"{name}/**/*.py"):
            p_program = str(p_program)
            p_program_sample_out = f"{p_program}_sample.out"
            if language == "Python":
                run_str = f"python {p_program} < {p_sample_in} > {p_program_sample_out}"
            try:
                subprocess.run(run_str, shell=True, capture_output=True, check=False)
            except:
                continue
            if not os.path.exists(p_program_sample_out):
                print(f"{p_program} did not generate any output, continuing to the next program.")
                continue
            with open(p_program_sample_out, "r") as f_program_sample_out:
                program_out = f_program_sample_out.readlines()
            if len(program_out) == 0:
                continue
            assert(len(program_out) == len(sample_out))

            good = 0
            for i in range(len(sample_out)):
                if program_out[i] == sample_out[i]:
                    good += 1
    
            if good / len(sample_out) > best_score:
                best_program = p_program
                best_score = good / len(sample_out)
        if not best_program:
            print(f"No valid code solutions for problem {name}.")
            results[name] = 0 / len(out)
            continue
        p_program_out = f"{p_program}.out"
        if language == "Python":
            run_str = f"python {p_program} < {p_in} > {p_program_out}"
        try:
            subprocess.run(run_str, shell=True, capture_output=True, check=False)
        except:
            print(f"No valid code solutions for problem {name}.")
            results[name] = 0 / len(out)
            continue

        with open(p_program_out, "r") as f_program_out:
            program_out = f_program_out.readlines()
        if len(program_out) == 0:
            continue
        assert(len(program_out) == len(out))

        good = 0
        for i in range(len(out)):
            if program_out[i] == out[i]:
                good += 1
        results[name] = good / len(out)


    print("| Problem | Score |")
    print("| ------- | ----- |")
    for name in sorted(list(results.keys())):
        print(f"| {name} | {results[name]} |")
    return results


def load_hf_model(model_args):
    kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **kwargs)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )
    tokenizer.pad_token = tokenizer.eos_token
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, config

def load_hf_data(
    data_args, model_args, training_args, tokenizer, config, require_text_cols=["solution"],
):
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=data_args.data_cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            train_split = "full" if "full" in raw_datasets.keys() else "train"
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"{train_split}[:{data_args.validation_split_percentage}%]",
                cache_dir=data_args.data_cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"{train_split}[{data_args.validation_split_percentage}%:]",
                cache_dir=data_args.data_cache_dir,
            )

    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        elif extension == "jsonl":
            extension = "json" 

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=data_args.data_cache_dir,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys() and training_args.do_eval:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=data_args.data_cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=data_args.data_cache_dir,
                token=model_args.token,
                **dataset_args,
            )

    column_names = list(raw_datasets["train"].features)
    # remove examples with missing fields
    raw_datasets = raw_datasets.filter(
        lambda example: all([example[c] is not None for c in data_args.text_cols])
    )
    raw_datasets = filter_examples(
        raw_datasets, require_text_cols=data_args.text_cols, 
        require_input_output=True, only_use_single_line_samples=True
        )

    if data_args.train_obj == 'lm': 
        def tokenize_function(examples):
            return tokenizer(format_example(examples, data_args.text_cols)[0])
        
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )
        
        block_size = min(data_args.block_size, tokenizer.model_max_length, config.max_position_embeddings or DEFAULT_BLOCK_SIZE)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        raw_datasets = tokenized_datasets.map(group_texts, batched=True)

    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    
    return train_dataset, eval_dataset


def train_hf_model(model, tokenizer, model_args, training_args, data_args, train_dataset, eval_dataset):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if data_args.train_obj == 'seq2seq':
        tokenizer.padding_side = 'right'
        response_template = "\n### Code:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

        def format_wrapper(examples):
            inputs, _ = format_example(examples, text_cols=data_args.text_cols)
            return inputs


        max_seq_len = (
            model.config.max_position_embeddings or MAX_CONTEXT_LENGTH
            if hasattr(model.config, "max_position_embeddings") and 
            model.config.max_position_embeddings is not None 
            else MAX_CONTEXT_LENGTH
        )

        trainer = SFTTrainer(
            model,
            tokenizer=tokenizer,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            args=training_args,
            formatting_func=format_wrapper,
            data_collator=collator,
            max_seq_length=max_seq_len,
        )

    elif data_args.train_obj == 'lm':
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def lm_compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

        compute_metrics = lm_compute_metrics if data_args.train_obj == 'lm' else None
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_xla_available()
            else None,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name
    trainer.create_model_card(**kwargs)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ProgramGenArguments, TrainingArguments))
    model_args, data_args, pg_args, training_args = parser.parse_args_into_dataclasses()

    if pg_args.only_run_generated_code:
        hackercupai_ds = load_dataset("hackercupai/hackercup", cache_dir="datasets")['full']
        hackercupai_ds = filter_examples(hackercupai_ds, require_text_cols=data_args.text_cols, require_input_output=True, only_use_single_line_samples=True)
        prompts, names = format_example(hackercupai_ds, text_cols=data_args.text_cols, include_output=False, language="Python")
        evaluate_programs(names, language=pg_args.gen_lang)
        return

    set_seed(training_args.seed)

    model, tokenizer, config = load_hf_model(model_args)

    train_dataset, eval_dataset = load_hf_data(data_args, model_args, training_args, tokenizer, config)

    if training_args.do_train:
        train_hf_model(model, tokenizer, model_args, training_args, data_args, train_dataset, eval_dataset)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    hackercupai_ds = load_dataset("hackercupai/hackercup", cache_dir="datasets")['full']
    hackercupai_ds = filter_examples(hackercupai_ds, require_text_cols=data_args.text_cols, require_input_output=True, only_use_single_line_samples=True)
    prompts, names = format_example(hackercupai_ds, text_cols=data_args.text_cols, include_output=False, language="Python")
    write_programs(
        prompts, names, pipeline, tokenizer, 
        max_time=pg_args.max_time, max_new_tokens=pg_args.max_new_tokens,
        num_gens_per_prompt=pg_args.num_gens, max_code_gen_examples=pg_args.max_code_gen_examples,
        language=pg_args.gen_lang, programs_path=pg_args.programs_path,
    )
    if pg_args.run_generated_code:
        evaluate_programs(names, language=pg_args.gen_lang)
    else:
        print("Generated programs saved to:", pg_args.programs_path)
        print("Please manually check the generated programs for malicious code and then "
              "run this script with --only_run_generated_code to evaluate the programs without re-generating them.")


if __name__ == "__main__":
    main()
