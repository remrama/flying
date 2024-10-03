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


def load_gpt_lucidity_codes(dataset):
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


def remove_short_and_long_dreams(df):
    lengths = df["dream_text"].str.len()
    return df[lengths.ge(50) & lengths.le(5000)]


def clean_dream_column(ser):
    return (
        ser.apply(unidecode.unidecode, errors="ignore", replace_str=None)
        .str.replace('"', "'")
        .str.strip()
    )


def load_dreamviews():
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


def load_sddb():
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
    dreams_only,
    name="Flying Dreams Database.xlsx",
    index_col="dream_ID",
    usecols=[
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
):
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


def save_json(obj: dict, filepath: str, mode: str = "wt", **kwargs):
    kwargs = {"indent": 4, "sort_keys": False, "ensure_ascii": True} | kwargs
    with open(filepath, mode, encoding="utf-8") as f:
        json.dump(obj, f, **kwargs)


def load_txt(filepath: str) -> str:
    """Load a raw text file as a string."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_matplotlib_settings(interactive=False):
    plt.rcParams["interactive"] = interactive
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.bf"] = "Arial:bold"
