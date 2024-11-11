"""Can ChatGPT identify lucidity?"""

import json
import os
from pathlib import Path
from time import sleep

import openai
import pandas as pd
from tqdm import tqdm

import utils


#######################################################################################
# Set OpenAI/ChatGPT model parameters
#######################################################################################

# Set OpenAI API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_KEY is not None, "OPENAI_API_KEY environment variable must be set."
openai.api_key = OPENAI_KEY

# List of tasks we set up for ChatGPT to perform
GPT_TASKS = ["annotate", "isdream", "islucid", "themesD", "themesM", "themesT"]

# OpenAI/ChatGPT model parameters
GPT_KWARGS = {
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


def load_task_prompt(target: str, task: str) -> str:
    """
    Load a custom prompt from a text file.

    Parameters
    ----------
    target : str
        The target type, must be either 'system' or 'user'.
    task : str
        The task name, must be one of the predefined GPT_TASKS.

    Returns
    -------
    str
        The content of the prompt file as a string.

    Raises
    ------
    AssertionError
        If `target` is not 'system' or 'user'.
        If `task` is not in GPT_TASKS.
        If the prompt does not contain exactly one '<INSERT_DREAM>' placeholder.

    Notes
    -----
    The function constructs the file path based on the `target` and `task` parameters
    and reads the content of the corresponding text file. The prompt must contain
    exactly one '<INSERT_DREAM>' placeholder.
    """
    assert target in ["system", "user"], "target must be 'system' or 'user'."
    assert task in GPT_TASKS, f"task must be one of {GPT_TASKS}."
    filepath = f"./prompt-{target}_task-{task}.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        prompt = f.read()
    assert prompt.count("<INSERT_DREAM>") == 1, "Prompt must contain 1 '<INSERT_DREAM>' match."
    return prompt


def load_json(filepath: str) -> dict:
    """
    Load a JSON file and return its contents as a dictionary.

    Parameters
    ----------
    filepath : str
        The path to the JSON file to be loaded.

    Returns
    -------
    dict
        The contents of the JSON file as a dictionary.

    Examples
    --------
    >>> data = load_json('path/to/file.json')
    >>> print(data)
    {'key': 'value'}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: dict, filepath: str, mode: str = "wt", **kwargs) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters
    ----------
    obj : dict
        The dictionary to be written to the JSON file.
    filepath : str
        The path to the file where the JSON data will be written.
    mode : str, optional
        The mode in which the file is opened. Default is 'wt' (write text).
    **kwargs : dict, optional
        Additional keyword arguments to pass to `json.dump`. Default values are:
        - indent: 4
        - sort_keys: False
        - ensure_ascii: True
    """
    kwargs = {"indent": 4, "sort_keys": False, "ensure_ascii": True} | kwargs
    with open(filepath, mode, encoding="utf-8") as f:
        json.dump(obj, f, **kwargs)



def main(dreams: pd.Series, task: str, model_kwargs: dict, overwrite: bool) -> None:
    """
    Perform a ChatGPT task on a series of dream reports.

    Parameters
    ----------
    dreams : pd.Series
        A pandas Series containing dream reports.
    task : str
        The specific task to be performed by ChatGPT.
    model_kwargs : dict
        A dictionary of keyword arguments to be passed to the ChatGPT model.
    overwrite : bool
        If True, overwrite existing responses; otherwise, load existing responses if available.
    """

    # Load the ChatGPT prompt text
    system_prompt = load_task_prompt("system", task)
    user_prompt = load_task_prompt("user", task)

    # Set the export path for the OpenAI/ChatGPT responses
    export_path = Path(utils.DERIV_DIR) / f"task-{task}_responses.json"
    export_path.parent.mkdir(exist_ok=True)

    # Load existing responses if they exist.
    if export_path.exists() and not overwrite:
        responses = load_json(export_path)
    else:
        # Initialize an empty dictionary to hold OpenAI responses/completions/results
        responses = {}

    # Initialize the ChatGPT messages.
    system_message = dict(role="system", content=system_prompt)
    user_message = dict(role="user")

    # Iterate over the dream reports and ask ChatGPT to perform the requested task
    for dream_id, dream_report in tqdm(dreams.items(), total=dreams.size, desc=f"GPT {task} task"):
        if dream_id not in responses:
            # Add this dream report to the ChatGPT prompt
            user_content = user_prompt.replace("<INSERT_DREAM>", dream_report)
            # Update ChatGPT model parameters with the updated user prompt
            user_message.update(content=user_content)
            model_kwargs.update(messages=[system_message, user_message])
            # Request a response from OpenAI/ChatGPT
            move_on = False
            while not move_on:
                try:
                    responses[dream_id] = openai.ChatCompletion.create(**model_kwargs)
                    move_on = True
                except openai.error.RateLimitError:
                    print("Rate Limit Error, backing off and trying again...")
                    sleep(1.0)
            # Write cumulative results to file
            write_json(responses, export_path)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", required=True, type=str, choices=GPT_TASKS, help="Task for ChatGPT to perform.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it already exists.")
    parser.add_argument("--test", action="store_true", help="Test by running on only 10 samples.")
    args = parser.parse_args()

    task = args.task
    overwrite = args.overwrite
    testing = args.test

    # Load Flying dreams as a pandas Series
    nfc = utils.load_sourcedata(dreams_only=True)
    if testing:
        nfc = nfc.sample(n=10, random_seed=32)
    assert nfc.index.name == "dream_id", "Index must be 'dream_id'."
    assert nfc.index.is_unique, "Index must contain all unique values."
    dreams = nfc["dream_text"]

    main(dreams, task, GPT_KWARGS, overwrite)
