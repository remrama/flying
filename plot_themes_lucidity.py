"""Evaluate performance of ChatGPT at coding for predetermined themes."""

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import utils


# Load custom matplotlib settings.
utils.load_matplotlib_settings(interactive=True)

# Choose the dataset and task.
dataset = "flying"
task = "thematicT"

# Load the data.
df = utils.load_sourcedata(dreams_only=True).drop(columns="GPT_ID500")

# Define the import paths.
import_name_responses = f"data-{dataset}_task-{task}_responses.json"
import_path_responses = utils.deriv_dir / import_name_responses

# Define the themes for each task.
if task.endswith("T"):
    themes = [
        "wings",
        "hovering/levitation",
        "running",
        "swimming-like movements",
        "spinning/rotation",
        "wind",
        "falling forward/launching from a high place",
        "focus/concentration",
        "jetpacks/rockets/suits",
        "balloons",
        "breath-related flying",
        "jumping/bouncing",
        "sorcery",
        "flying objects",
        "flying beings",
        "flying vehicles",
        "transformation",
        "climbing/stepping in the air",
        "superhero",
        "unspecified",
    ]
    short_names = {
        "hovering/levitation": "hovering",
        "swimming-like movements": "swimming",
        "spinning/rotation": "spinning",
        "falling forward/launching from a high place": "falling",
        "focus/concentration": "focus",
        "jetpacks/rockets/suits": "jetpacks",
        "breath-related flying": "breath",
        "jumping/bouncing": "jumping",
        "flying objects": "objects",
        "flying beings": "beings",
        "flying vehicles": "vehicules",
        "climbing/stepping in the air": "climbing",
    }
elif task.endswith("M"):
    themes = [
        "in response to fear",
        "enjoyment",
        "learning/practice",
        "mean of transportation",
        "involuntary flight",
        "elicit a reaction in other people",
        "helping/saving others",
        "reality check",
        "unspecified",
    ]
    short_names = {
        "in response to fear": "fear",
        "enjoyment": "fun",
        "learning/practice": "learning",
        "mean of transportation": "transport",
        "involuntary flight": "unvoluntary",
        "elicit a reaction in other people": "elicit reaction",
        "helping/saving others": "helping",
        "reality check": "rc",
        "unspecified": "not specified",
    }
elif task.endswith("D"):
    themes = [
        "bodily/physical limitations",
        "environmental constraints",
        "fear/anxiety",
        "lack of belief",
        "lack of focus",
        "waking up",
        "technical failure",
        "other beings",
        "gravity/crash/falling",
        "restricted speed/altitude",
        "inability to initiate flight",
        "no obstacles",
    ]
    short_names = {
        "bodily/physical limitations": "body",
        "environmental constraints": "environment",
        "fear/anxiety": "fear",
        "lack of belief": "belief",
        "lack of focus": "focus",
        "waking up": "wake",
        "technical failure": "tech",
        "other beings": "beings",
        "gravity/crash/falling": "crash",
        "restricted speed/altitude": "slow",
        "inability to initiate flight": "inability",
        "no obstacles": "no obstacles",
    }

# Load the responses/completions from ChatGPT.
completions = utils.load_json(import_path_responses)
results = {}
for dream_id, completion in completions.items():
    choices = completion["choices"]
    assert len(choices) == 1, "Expected only 1 response from ChatGPT."
    choice = choices[0]
    assert choice["finish_reason"] == "stop", "Expected stop as the finish reason."
    content = choice["message"]["content"]
    assert content.startswith("{") and content.endswith("}"), "Expected JSON output."
    ann = json.loads(content)
    # Sometimes ChatGPT adds an extra theme, so remove it (ITS SO RARE, like once?)
    ann = {k: v for k, v in ann.items() if k in themes}
    assert len(ann.keys()) == len(themes)
    assert all(k in themes for k in ann)
    assert all(isinstance(v, bool) for v in ann.values())
    results[dream_id] = ann

# Convert to a dataframe with themes as columns and cells as 1 or 0.
scores = (
    pd.DataFrame.from_dict(results, orient="index")
    .rename_axis("dream_id")
    .rename(columns=short_names)
    # .rename(columns={"vehicules": "vehicles"})
    .astype(int)
)

# Load GPT's lucidity scores.
ser = utils.load_gpt_lucidity_codes(dataset="flying")

# Merge GPT scores with dataset.
dat = scores.join(ser, how="inner")

# Get the frequencies of each theme by lucidity.
freqs = dat.groupby("lucidity").mean().T
freqs.index = freqs.index.str.replace("vehicules", "vehicles")
freqs["summ"] = freqs["lucid"] + freqs["non-lucid"]
freqs = freqs.drop("focus")
freqs = freqs.drop("unspecified")
freqs = freqs.sort_values("summ", ascending=False)
freqs = freqs.head(10)
freqs = freqs.sort_values("non-lucid", ascending=False)
freqs = freqs.drop(columns="summ")

# Melt the data for plotting.
melted = freqs.stack().reset_index().rename(columns={"level_0": "theme", 0: "freq"})

# Plot the data.
fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
sns.barplot(
    data=melted,
    x="theme",
    y="freq",
    hue="lucidity",
    saturation=1,
    palette=utils.colors,
    hue_order=["non-lucid", "lucid"],
    order=freqs.index.tolist(),
    ax=ax,
)

# Set ticks and labels.
ax.set_xticklabels(freqs.index.tolist(), rotation=40, ha="right")
ax.set_xlabel("Flying technique")
ax.set_ylabel("Percentage of dreams")
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.02))
ax.set_ybound(upper=0.40)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))

# Save the plot.
export_path = utils.deriv_dir / f"data-{dataset}_themes-techniq_lucidity.png"
plt.savefig(export_path)
plt.close()
