"""
Prepare a dataset for fine-tuning an OpenAI model.
"""
import argparse
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filepath", type=str, default="./dreams.tsv", help="Full filepath of data to load.")
parser.add_argument("-p", "--prompt", type=str, default="raw", help="Column holding prompts.")
parser.add_argument("-c", "--completion", type=str, default="clean", help="Column holding completions.")
args = parser.parse_args()

filepath = args.filepath
prompt_column = args.prompt
completion_column = args.completion


# Depending on the model used, requests can use up to 4097 tokens shared between prompt and completion.
# If your prompt is 4000 tokens, your completion can be 97 tokens at most. 
max_prompt_tokens = 2000

filepath = Path(filepath).expanduser()
export_path = filepath.with_suffix(".jsonl")

read_kwargs = {"usecols": [prompt_column, completion_column]}

if (suffix := filepath.suffix) == ".tsv":
    df = pd.read_table(filepath, **read_kwargs)
elif suffix == ".csv":
    df = pd.read_csv(filepath, **read_kwargs)
else:
    raise IOError("Expects tsv or csv file as input.")

df = df.rename(columns={prompt_column: "prompt", completion_column: "completion"})

df = df[df["prompt"].str.len() <= max_prompt_tokens]

# Check that the separator we intend to use isn't present within the contexts
assert not df["prompt"].str.contains("\n\n###\n\n").any()
assert not df["completion"].str.contains("END").any()
df["prompt"] = df["prompt"].str.strip().add("\n\n###\n\n")
df["completion"] = df["completion"].str.strip().radd(" ").add(" END")

df = df[:-1]

df.to_json(export_path, orient="records", lines=True)
