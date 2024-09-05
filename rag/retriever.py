import ast
import logging
import os
from pathlib import Path
from typing import List, Optional

import bm25s
import pandas as pd
import weave
from datasets import load_dataset
from joblib import Parallel, delayed
from openai import AsyncOpenAI
from simple_parsing import ArgumentParser
from sklearn.metrics.pairwise import cosine_similarity

from utils import (EMBEDDING_MODEL, Problem, Solution, clean_code_string,
                   remove_extra_newlines)

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

# Data Loading

LANGUAGE_MAP = {
    3: "Python3",
}


def clean_code(row: dict) -> dict:
    outputs = []
    for item in row["code"]:
        item = clean_code_string(item)
        outputs.append(item)
    return {"code": outputs}


def get_solution(row: dict) -> dict:
    solutions = row["solutions"]
    languages = solutions["language"]
    solutions = solutions["solution"]

    outputs = []
    for language, solution in zip(languages, solutions):
        language = LANGUAGE_MAP.get(language)
        if language:
            outputs.append(solution)
    return {"code": outputs}


def get_test_cases(row: dict) -> dict:
    tests = row["public_tests"]
    return {
        "sample_inputs": "".join(tests["input"]),
        "sample_outputs": "".join(tests["output"]),
    }


def clean_description(row: dict) -> dict:
    description = row["description"]
    description = remove_extra_newlines(description)
    return {"description": description}


def get_code_contests_data(cache_file: Path, reload_cache: bool = False):
    if cache_file.exists() and not reload_cache:
        logger.info(f"Loading cached raw data from {cache_file}")
        return pd.read_json(cache_file, lines=True)

    logger.info(f"Loading raw data from dataset")
    ds = load_dataset("deepmind/code_contests")

    train_ds = ds["train"].map(get_solution, num_proc=4)
    train_ds = train_ds.filter(lambda x: not x["is_description_translated"], num_proc=4)
    train_ds = train_ds.filter(lambda x: len(x["code"]) > 0, num_proc=4)
    train_ds = train_ds.map(clean_code, num_proc=4)
    train_ds = train_ds.map(clean_description, num_proc=4)
    train_ds = train_ds.map(get_test_cases, num_proc=4)
    train_ds = train_ds.remove_columns(
        [
            col
            for col in train_ds.column_names
            if col not in ["description", "code", "sample_inputs", "sample_outputs"]
        ]
    )

    train_df = train_ds.to_pandas()
    train_df = train_df.explode("code").reset_index(drop=True)
    train_df = train_df.drop_duplicates(subset=["code"], keep="first")
    train_df.to_json(cache_file, orient="records", lines=True)
    return train_df


# Data Preprocessing

# Define a mapping from AST node types to token names
TOKEN_MAP = {
    ast.FunctionDef: "FUNC_DEF",
    ast.ClassDef: "CLASS_DEF",
    ast.BinOp: "BIN_OP",
    ast.Assign: "ASSIGN",
    ast.Expr: "EXPR",
    ast.Call: "FUNC_CALL",
    ast.If: "IF",
    ast.For: "FOR",
    ast.While: "WHILE",
    ast.Import: "IMPORT",
    ast.Return: "RETURN",
    ast.List: "LIST",
    ast.Dict: "DICT",
    ast.Name: "VAR",
    ast.Num: "NUMBER",  # For older Python versions (< 3.8)
    ast.Constant: lambda node: (
        "NUMBER"
        if isinstance(node.value, (int, float, complex))
        else (
            "STRING"
            if isinstance(node.value, str)
            else (
                "BOOLEAN"
                if isinstance(node.value, bool)
                else "NONE" if node.value is None else "UNKNOWN"
            )
        )
    ),
}


def tokenize_node(node):
    """Tokenizes an AST node using the TOKEN_MAP dictionary."""
    node_type = type(node)

    # Handle the case where the node type is in the TOKEN_MAP
    if node_type in TOKEN_MAP:
        token = TOKEN_MAP[node_type]
        if callable(
            token
        ):  # If the token is a function (for complex cases like ast.Constant)
            yield token(node)
        else:
            yield token

    # Recursively process child nodes
    for child in ast.iter_child_nodes(node):
        yield from tokenize_node(child)


def normalize_code(code: str) -> Optional[str]:
    """Tokenizes and normalizes any Python code snippet."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return None

    tokens = list(tokenize_node(tree))
    return " ".join(tokens)


def normalize_code_list(code_list: list[str]) -> list[str]:
    if len(code_list) > 1000:
        return Parallel(n_jobs=-1)(delayed(normalize_code)(code) for code in code_list)
    else:
        return [normalize_code(code) for code in code_list]


def preprocess_data(
    input_path: Path, output_path: Path, reload_cache: bool = False
) -> pd.DataFrame:
    if output_path.exists() and not reload_cache:
        logger.info(f"Loading cached preprocessed data from {output_path}")
        return pd.read_json(output_path, lines=True)

    logger.info(f"Preprocessing data from {input_path}")
    data_df = pd.read_json(input_path, lines=True)
    data_df["normalized_code"] = normalize_code_list(data_df["code"].tolist())
    data_df = data_df.dropna(subset=["normalized_code"])
    data_df.to_json(output_path, orient="records", lines=True)
    return data_df


class Retriever:
    def __init__(self, path: str = "param-bharat/rag-hackercup"):
        ds = load_dataset(path, split="train")
        data_df = ds.to_pandas()
        self.docs = data_df.to_dict(orient="records")
        self.corpus = data_df["normalized_code"]
        self.retriever = self.index()

    def index(self):
        corpus = self.corpus.tolist()
        corpus_tokens = bm25s.tokenize(corpus, stopwords=None)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        return retriever

    @weave.op(name="retrieve_docs")
    def retrieve(self, query: str, k: int = 10):
        clean_query = clean_code_string(query)
        normalized_query = normalize_code(clean_query)
        query_tokens = bm25s.tokenize(normalized_query, stopwords=None)
        docs, _ = self.retriever.retrieve(query_tokens, k=k, corpus=self.docs)
        return docs[0, :].tolist()


def index_data(
    input_path: Path,
    output_path: Path,
    reload_cache: bool = False,
):
    if output_path.exists() and not reload_cache:
        logger.info(f"Loading cached retriever from {output_path}")
        return Retriever.load(output_path)
    logger.info(f"Creating retriever from {input_path}")
    data_df = pd.read_json(input_path, lines=True, orient="records")
    retriever = Retriever(data_df=data_df)
    retriever.index()
    retriever.save(output_path)
    return retriever


@weave.op(name="get_embeddings")
async def get_embeddings(texts, model=EMBEDDING_MODEL):
    async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if isinstance(texts, str):
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]
    response = await async_client.embeddings.create(
        input=texts, model=model, dimensions=512
    )
    return [embedding.embedding for embedding in response.data]


@weave.op(name="rerank_docs")
async def rerank_docs(
    problem: Problem,
    solution: Solution,
    retrieved_docs: List[dict],
    top_k: int = 3,
) -> List[dict]:
    query_embeddings = await get_embeddings(
        problem.problem_description + " " + solution.source_code
    )
    docs_embeddings = await get_embeddings(
        [doc["description"] + " " + doc["code"] for doc in retrieved_docs]
    )

    similarities = cosine_similarity(query_embeddings, docs_embeddings)
    docs_df = pd.DataFrame(retrieved_docs)
    docs_df["similarity"] = similarities[0]
    docs_df = docs_df.sort_values(by="similarity", ascending=False)
    docs_df = docs_df.drop_duplicates(
        subset=["description"],
        keep="first",
    )
    return docs_df.head(top_k).to_dict(orient="records")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-c", "--cache-directory", type=Path, default="data/cache")
    parser.add_argument("--reload-cache", action="store_true")

    args = parser.parse_args()

    if not args.cache_directory.exists():
        args.cache_directory.mkdir(parents=True)

    if (args.cache_directory / "retriever").exists():
        retriever = Retriever.load(args.cache_directory / "retriever")
    elif (args.cache_directory / "preprocessed.jsonl").exists():
        preprocessed_df = preprocess_data(
            args.cache_directory / "raw.jsonl",
            args.cache_directory / "preprocessed.jsonl",
            args.reload_cache,
        )
        retriever = Retriever(data_df=preprocessed_df)
        retriever.index()
        retriever.save(args.cache_directory / "retriever")
    else:
        raw_df = get_code_contests_data(
            args.cache_directory / "raw.jsonl", args.reload_cache
        )
        preprocessed_df = preprocess_data(
            args.cache_directory / "raw.jsonl",
            args.cache_directory / "preprocessed.jsonl",
            args.reload_cache,
        )
        retriever = Retriever(data_df=preprocessed_df)
        retriever.index()
        retriever.save(args.cache_directory / "retriever")
