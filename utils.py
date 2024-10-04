"""Utility functions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import unidecode
import yaml


SOURCE_DIR = "../sourcedata"
DERIV_DIR = "../derivatives"

source_dir = Path(SOURCE_DIR).expanduser()
deriv_dir = Path(DERIV_DIR).expanduser()

colors = {
    "dream": "#1E71B5",
    "comment": "gainsboro",
    "lucidity": "orange",
    "flying": "#1E71B5",
    "supplement": "gainsboro",
    "lucid": "orange",
    "non-lucid": "#6CADD5",
    "female": "rebeccapurple",
    "male": "mediumpurple",
    "they": "blueviolet",
    "unspecified": "gainsboro",
}


def load_gpt_lucidity_codes(dataset: str) -> pd.Series:
    """
    Load GPT-generated lucidity codes for a given dataset.
    This function asserts that the provided dataset is one of the allowed values 
    ("dreamviews", "flying", "sddb"). It then loads the corresponding JSON file 
    containing GPT completions and processes the data to extract lucidity codes 
    for each dream entry.
    Args:
        dataset (str): The name of the dataset to load. Must be one of 
                       ["dreamviews", "flying", "sddb"].
    Returns:
        pd.Series: A pandas Series with dream IDs as the index and lucidity 
                   status ("lucid" or "non-lucid") as the values.
    """
    
    assert dataset in ["dreamviews", "flying", "sddb"]
    responses = {}
    completions = load_json(deriv_dir / f"data-{dataset}_task-islucid_responses.json")
    for dream_id, completion in completions.items():
        responses[dream_id] = [
            choice["message"]["content"] for choice in completion["choices"]
        ]
    ser = (
        pd.Series(responses)
        .explode()
        .rename("lucidity")
        .rename_axis("dream_id")
        .map({"True": "lucid", "False": "non-lucid"})
    )
    return ser


def remove_short_and_long_dreams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out rows in the DataFrame where the length of the 'dream_text' column
    is less than 50 characters or greater than 5000 characters.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing a 'dream_text' column.
    Returns:
    pandas.DataFrame: A DataFrame with rows filtered based on the length of 'dream_text'.
    """
    
    lengths = df["dream_text"].str.len()
    return df[lengths.ge(50) & lengths.le(5000)]


def clean_dream_column(ser: pd.Series) -> pd.Series:
    """
    Cleans a pandas Series containing dream-related text data.
    This function performs the following operations on the input Series:
    1. Applies the `unidecode` function to convert any non-ASCII characters to their closest ASCII equivalents.
    2. Replaces double quotes (") with single quotes (').
    3. Strips leading and trailing whitespace from each string in the Series.
    Args:
        ser (pd.Series): A pandas Series containing text data to be cleaned.
    Returns:
        pd.Series: A pandas Series with the cleaned text data.
    """
    
    return (
        ser.apply(unidecode.unidecode, errors="ignore", replace_str=None)
        .str.replace('"', "'")
        .str.strip()
    )


def load_dreamviews() -> pd.DataFrame:
    """
    Loads and processes the DreamViews dataset from a TSV file.
    The function performs the following steps:
    1. Reads the TSV file into a pandas DataFrame.
    2. Filters the DataFrame to include only rows where the 'lucidity' column is either 'lucid' or 'nonlucid'.
    3. Renames columns: 'post_id' to 'dream_id' and 'post_clean' to 'dream_text'.
    4. Sets 'dream_id' as the index of the DataFrame.
    5. Replaces 'nonlucid' with 'non-lucid' in the 'lucidity' column.
    6. Drops rows where 'dream_text' is NaN.
    7. Prefixes the 'dream_id' index with 'DV-'.
    8. Cleans the 'dream_text' column using the `clean_dream_column` function.
    9. Removes dreams that are too short or too long using the `remove_short_and_long_dreams` function.
    Returns:
        pandas.DataFrame: The processed DreamViews dataset.
    """
    
    import_path = source_dir / "dreamviews.tsv"
    df = pd.read_table(import_path)
    df = df[df["lucidity"].isin(["lucid", "nonlucid"])]
    df = (
        df.rename(columns={"post_id": "dream_id", "post_clean": "dream_text"})
        .set_index("dream_id")
        .replace({"lucidity": {"nonlucid": "non-lucid"}})
        .dropna(subset="dream_text")
    )
    df.index = df.index.map("DV-{}".format)
    df.loc[:, "dream_text"] = clean_dream_column(df["dream_text"])
    return remove_short_and_long_dreams(df)


def load_sddb() -> pd.DataFrame:
    """
    Loads the SDDb.csv file, processes the data, and returns a cleaned DataFrame.
    The function performs the following steps:
    1. Reads the SDDb.csv file located in the source directory.
    2. Selects specific columns: "answer_text", "dream_entry_title", "respondent", and "survey".
    3. Renames the "answer_text" column to "dream_text".
    4. Drops rows where "dream_text" is NaN.
    5. Sets the DataFrame index to a formatted string "SDDB-{index}".
    6. Cleans the "dream_text" column using the `clean_dream_column` function.
    7. Removes rows with short or long dreams using the `remove_short_and_long_dreams` function.
    Returns:
        pd.DataFrame: A cleaned DataFrame with processed dream data.
    """
    
    import_path = source_dir / "SDDb.csv"
    df = (
        pd.read_csv(
            import_path,
            usecols=["answer_text", "dream_entry_title", "respondent", "survey"],
            low_memory=False,
        )
        .rename(columns={"answer_text": "dream_text"})
        .dropna(subset="dream_text")
    )
    df.index = pd.Index([f"SDDB-{x:06d}" for x in range(len(df))], name="dream_id")
    df.loc[:, "dream_text"] = clean_dream_column(df["dream_text"])
    return remove_short_and_long_dreams(df)


def load_sourcedata(
    dreams_only: bool,
    name: str = "Flying Dreams Database.xlsx",
    index_col: str = "dream_ID",
    usecols: list = [
        "dream_ID",
        "source",
        "participant_ID",
        # "user_info",
        "sex",
        "date",
        "report_type",
        "thread_keywords",
        "dream",
        "GPT_ID500",
    ],
    **kwargs,
) -> pd.DataFrame:
    """
    Load and preprocess dream data from an Excel file.
    Parameters:
    - dreams_only (bool): If True, filter the data to include only dream reports.
    - name (str): The name of the Excel file to load. Default is "Flying Dreams Database.xlsx".
    - index_col (str): The column to use as the index. Default is "dream_ID".
    - usecols (list): List of columns to use from the Excel file. Default includes specific columns.
    - **kwargs: Additional keyword arguments to pass to `pd.read_excel`.
    Returns:
    - pd.DataFrame: A DataFrame containing the preprocessed dream data.
    """

    filepath = source_dir / name
    df = (
        pd.read_excel(filepath, index_col=index_col, usecols=usecols, **kwargs)[
            usecols[1:]
        ]
        .rename(
            columns={
                "thread_keywords": "title",
                "dream_ID": "dream_id",
                "source": "source_id",
                "participant_ID": "subject_id",
                "dream": "dream_text",
            }
        )
        .replace(
            {
                "report_type": {1: "dream", 2: "comment"},
                "sex": {
                    1: "male",
                    2: "female",
                    4: "they",
                    "Female": "female",
                    "unspecified": "unspecified",
                },
                "source_id": {
                    "Alchemy forums": "AlchemyForums",
                    "DreamBank": "DreamBank",
                    "Flying dreams_CPD_anonymised": "CPD",
                    "I Dream of Covid (IDoC)": "IDoC",
                    "International Archive of Dreams": "IAoD",
                    "LD4all.com : Dream Journal": "LD4all",
                    "LD4all.com : Lucid adventures": "LD4all",
                    "LD4all.com : Quest for Lucidity": "LD4all",
                    "LD4all.com :Lucid Adventures": "LD4all",
                    "LD4all.com : quest for lucidity": "LD4all",
                    "Reddit": "Reddit",
                    "Reddit R/dreams": "Reddit",
                    "Reddit r/ astralprojection": "Reddit",
                    "Reddit r/AstralProjection": "Reddit",
                    "Reddit r/AskReddit": "Reddit",
                    "Reddit r/Dream": "Reddit",
                    "Reddit r/DreamInterpretation": "Reddit",
                    "Reddit r/Dreams": "Reddit",
                    "Reddit r/ElectricSkateboarding": "Reddit",
                    "Reddit r/InDreams": "Reddit",
                    "Reddit r/LucidDreaming": "Reddit",
                    "Reddit r/Psychic": "Reddit",
                    "Reddit r/ShrugLifeSyndicate": "Reddit",
                    "Reddit r/Shittyaskflying": "Reddit",
                    "Reddit r/askReddit": "Reddit",
                    "Reddit r/askreddit": "Reddit",
                    "Reddit r/astralprojection": "Reddit",
                    "Reddit r/aviation": "Reddit",
                    "Reddit r/dreaminterpretation": "Reddit",
                    "Reddit r/dreams": "Reddit",
                    "Reddit r/flying": "Reddit",
                    "Reddit r/lucidDreaming": "Reddit",
                    "Reddit r/luciddreaming": "Reddit",
                    "Reddit r/mylittleandysonic1": "Reddit",
                    "Reddit r/mylittleandysonic2": "Reddit",
                    "Reddit r/polls": "Reddit",
                    "Reddit r/shittyaskflying": "Reddit",
                    "Reddit r/shruglifesyndicate": "Reddit",
                    "Reddit: r/LucidDreaming": "Reddit",
                    "Redditr r/LucidDreaming": "Reddit",
                    "Reddits r/LucidDreaming": "Reddit",
                    "SDDB Flying Dreams - Export": "SDDb",
                    "Straight Dope Message Board > Main > In My Humble Opinion (IMHO) >": "IMHO",
                    "The Lucidity Institute, Stephen Laberge": "LucidityInstitute",
                    "Tore Email": "Tore",
                    "TW_combined_final": "TW",
                    "Twitter : @CovidDreams 30mar2020_11jul2022 (33834) for Tobi v2": "Twitter",
                },
            }
        )
        .replace(
            {
                "source_id": {
                    "AlchemyForums": "Misc",
                    "CPD": "Misc",
                    "IDoC": "Misc",
                    "IAoD": "Misc",
                    "IMHO": "Misc",
                    "LucidityInstitute": "Misc",
                    "Tore": "Misc",
                    "TW": "Misc",
                    "Twitter": "Misc",
                }
            }
        )
        .fillna({"sex": "unspecified"})
        .sort_values(["source_id", "subject_id"])
        .dropna(subset="dream_text")
        .rename_axis("dream_id")
    )
    if dreams_only:
        df = df.query("report_type=='dream'")
    df.loc[:, "dream_text"] = clean_dream_column(df["dream_text"])
    return remove_short_and_long_dreams(df)


def load_config() -> dict:
    """Load YAML configuration file as a dictionary."""
    with open("./config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(filepath: str) -> dict:
    """Load JSON file as a dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, filepath: str, mode: str = "wt", **kwargs) -> None:
    """Save a dictionary as a JSON file."""
    kwargs = {"indent": 4, "sort_keys": False, "ensure_ascii": True} | kwargs
    with open(filepath, mode, encoding="utf-8") as f:
        json.dump(obj, f, **kwargs)


def load_txt(filepath: str) -> str:
    """Load a raw text file as a string."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_matplotlib_settings(interactive: bool = False) -> None:
    """Load custom matplotlib settings."""
    plt.rcParams["interactive"] = interactive
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.bf"] = "Arial:bold"
