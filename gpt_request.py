"""Can ChatGPT identify lucidity?"""

import argparse
import os
from time import sleep

import openai
from tqdm import tqdm

import utils


available_datasets = ["dreamviews", "flying", "sddb"]

available_tasks = [
    "isdream",
    "islucid",
    "annotate",
    "thematicD",
    "thematicM",
    "thematicT",
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset", required=True, type=str, choices=available_datasets
)
parser.add_argument("-t", "--task", required=True, type=str, choices=available_tasks)
parser.add_argument(
    "-o",
    "--overwrite",
    action="store_true",
    help="Overwrite output file if it already exists.",
)
parser.add_argument("--test", action="store_true", help="Just run on 10 samples.")
args = parser.parse_args()

dataset = args.dataset
overwrite = args.overwrite
task = args.task
testing = args.test


# Set OpenAI API key.
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load data
if dataset == "flying":
    df = utils.load_sourcedata(dreams_only=True)
elif dataset == "dreamviews":
    df = utils.load_dreamviews()
elif dataset == "sddb":
    df = utils.load_sddb()

# For testing, just use a small sample of the data.
if testing:
    df = df.sample(n=100, random_seed=32)

assert df.index.name == "dream_id"
assert df.index.is_unique

# Get the dream text from the DataFrame.
ser = df["dream_text"]

# Load the ChatGPT prompt text.
system_prompt = utils.load_txt(f"./prompt-system_task-{task}.txt")
user_prompt = utils.load_txt(f"./prompt-user_task-{task}.txt")

# Set the export path for the OpenAI responses.
export_path = utils.deriv_dir / f"data-{dataset}_task-{task}_responses.json"

# Set OpenAI/ChatGPT model parameters.
model_kwargs = {
    "model": "gpt-4",
    "temperature": 0,  # Lower means more deterministic results
    "top_p": 1,  # Also impacts determinism, but don't modify this and temperature
    "n": 1,  # Number of responses
    "stream": False,
    "stop": None,
    "max_tokens": None,  # The maximum number of tokens to generate in the chat completion
    "presence_penalty": 0,  # Penalizes tokens for occurring (or being absent if negative)
    "frequency_penalty": 0,
}

# Load existing responses if they exist.
if export_path.exists() and not overwrite:
    responses = utils.load_json(export_path)
else:
    # Initialize an empty dictionary to hold OpenAI responses/completions/results.
    responses = {}

# Initialize the ChatGPT messages.
system_message = dict(role="system", content=system_prompt)
user_message = dict(role="user")

# Iterate over the dream reports and ask ChatGPT to identify lucidity.
for dream_id, dream_report in tqdm(ser.items(), total=ser.size, desc="Dreams"):
    if dream_id not in responses:
        # Add this dream report to the ChatGPT prompt.
        user_content = user_prompt.replace("<INSERT_DREAM>", dream_report)
        # Update ChatGPT model parameters with the updated user prompt.
        user_message.update(content=user_content)
        model_kwargs.update(messages=[system_message, user_message])
        # Request a response from OpenAI/ChatGPT.
        move_on = False
        while not move_on:
            try:
                responses[dream_id] = openai.ChatCompletion.create(**model_kwargs)
                move_on = True
            except openai.error.RateLimitError:
                print("Rate Limit Error, backing off and trying again...")
                sleep(1.0)
        # Write cumulative results to file.
        utils.save_json(responses, export_path)
