"""Plot spans of lucidity and flying."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

import utils


# Load custom matplotlib settings.
utils.load_matplotlib_settings(interactive=True)

# Choose the dataset and task.
dataset = "flying"
task = "annotate"

# Define the import/export paths.
import_name = f"data-{dataset}_task-{task}_responses.json"
export_name = f"data-{dataset}_lucidVsFlying.png"
import_path = utils.deriv_dir / import_name
export_path = utils.deriv_dir / export_name

# Load the responses/completions from ChatGPT.
completions = utils.load_json(import_path)

# Set expected labels and normalization parameters.
expected_labels = ["flying", "lucidity", "supplement"]
norm_length = 100
norm_index = np.linspace(0, 1, num=norm_length)

# Initialize the results dictionary.
badcounts = 0
results = {}

# Iterate over the completions and extract the annotations.
for dream_id, completion in completions.items():
    try:
        choices = completion["choices"]
        assert len(choices) == 1, "Expected only 1 response from ChatGPT."
        choice = choices[0]
        assert choice["finish_reason"] == "stop"
        content = choice["message"]["content"]
        assert content.startswith("{") and content.endswith("}")
        ann = json.loads(content)
        assert len(ann.keys()) == 2
        assert all(k in ["text", "entities"] for k in ann)
        dream_report = ann["text"]
        entities = ann["entities"]
        assert isinstance(entities, list)
        if len(entities) > 0:
            unique_labels = set(e["label"] for e in entities)
            assert all(label in expected_labels for label in unique_labels)
            n_total_characters = len(dream_report)
            masks = {
                label: np.zeros(n_total_characters, dtype=int) for label in expected_labels
            }
            for e in entities:
                entity_text = e["value"]
                entity_label = e["label"]
                try:
                    start = dream_report.index(entity_text)
                except ValueError:
                    print("wrong value")
                    continue
                end = start + len(entity_text)
                window = np.arange(start, end)
                masks[entity_label][window] = 1
            old_index = np.linspace(0, 1, num=n_total_characters)
            masks = {k: np.interp(norm_index, old_index, v) for k, v in masks.items()}
            # masks = {k: interp1d(old_index, v, kind="nearest")(norm_index) for k, v in masks.items()}
            results[dream_id] = masks
    except:
        badcounts += 1

print(f"THIS MANY LOADING ERRORS: {badcounts}")

# Convert to a dataframe with themes as columns and cells as 1 or 0.
flying = np.stack([v["flying"] for v in results.values()])
supplement = np.stack([v["supplement"] for v in results.values()])
lucidity = np.stack([v["lucidity"] for v in results.values()])


################################################################################
# Visualize the timecourse of supplements and flying labels.
################################################################################

# Define plotting parameters.
plot_kwargs = dict(linestyle="solid", linewidth=1, alpha=1)
fbetween_kwargs = dict(linewidth=0, alpha=0.3)
palette = utils.colors.copy()
zorder = dict(flying=3, lucidity=2, supplement=1)
x = np.arange(norm_length)

# Create the plot.
fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
for label, data in zip(["supplement", "flying"], [supplement, flying]):
    mean = data.mean(axis=0)
    sem = stats.sem(data, axis=0)
    color = palette[label]
    z = zorder[label]
    ax.fill_between(x, mean - sem, mean + sem, color=color, zorder=z, **fbetween_kwargs)
    ax.plot(x, mean, color=color, label=label, zorder=z, **plot_kwargs)

# Set the plot labels and limits.
ax.set_ylabel("Percentage of dreams")
ax.set_xlabel(
    "Progression of the dream\n"
    r"Start $\leftarrow$                           $\rightarrow$ End"
)
ax.set_xlim(0, 100)
ax.xaxis.set(
    major_locator=plt.MultipleLocator(25), minor_locator=plt.MultipleLocator(5)
)
ax.set_ylim(0, 0.3)
ax.yaxis.set(
    major_locator=plt.MultipleLocator(0.1), minor_locator=plt.MultipleLocator(0.02)
)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))

# Add the legend.
legend = ax.legend(
    loc="lower left",
    title="Event",
    frameon=False,
    title_fontsize=8,
    fontsize=8,
    handletextpad=0.4,  # space between legend marker and label
    labelspacing=0.1,  # vertical space between the legend entries
    borderaxespad=0,
)
legend._legend_box.align = "left"
# legend._legend_box.sep = 5 # brings title up farther on top of handles/labels

# Save the plot.
export_path = utils.deriv_dir / f"data-{dataset}_task-{task}_supp.png"
plt.savefig(export_path)
plt.close()


################################################################################
# Visualize the timecourse of supplements and lucidity labels.
################################################################################

fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)

plot_kwargs = dict(linestyle="solid", linewidth=1, alpha=1)
fbetween_kwargs = dict(linewidth=0, alpha=0.3)
palette = utils.colors.copy()
zorder = dict(flying=3, lucidity=2, supplement=1)
x = np.arange(norm_length)

idx = lucidity.max(axis=1) > 0
for label, data in zip(["lucidity", "flying"], [lucidity[idx], flying[idx]]):
    mean = data.mean(axis=0)
    sem = stats.sem(data, axis=0)
    color = palette[label]
    z = zorder[label]
    ax.fill_between(x, mean - sem, mean + sem, color=color, zorder=z, **fbetween_kwargs)
    ax.plot(x, mean, color=color, label=label, zorder=z, **plot_kwargs)

ax.set_ylabel("Percentage of dreams")
ax.set_xlabel(
    "Progression of the dream\n"
    r"Start $\leftarrow$                           $\rightarrow$ End"
)
ax.set_xlim(0, 100)
ax.xaxis.set(
    major_locator=plt.MultipleLocator(25), minor_locator=plt.MultipleLocator(5)
)

ax.set_ylim(0, 0.3)
ax.yaxis.set(
    major_locator=plt.MultipleLocator(0.1), minor_locator=plt.MultipleLocator(0.02)
)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))

legend = ax.legend(
    loc="upper left",
    title="Event",
    frameon=False,
    title_fontsize=8,
    fontsize=8,
    handletextpad=0.4,  # space between legend marker and label
    labelspacing=0.1,  # vertical space between the legend entries
    borderaxespad=0,
)

legend._legend_box.align = "left"
# legend._legend_box.sep = 5 # brings title up farther on top of handles/labels

export_path = utils.deriv_dir / f"data-{dataset}_task-{task}_lucidity.png"
plt.savefig(export_path)
plt.close()


################################################################################
# heatmap of lucid vs flying?
################################################################################

flying_onsets = np.argmax(flying[idx], axis=1)
lucidity_onsets = np.argmax(lucidity[idx], axis=1)
onsets = pd.DataFrame({"flying": flying_onsets, "lucidity": lucidity_onsets})

bar_kwargs = dict(lw=1, edgecolor="black", height=0.7)
errorbar_kwargs = dict(fmt="-o", color="black", lw=1, markersize=3)
fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
order = ["flying", "lucidity"]

means = onsets.mean().loc[order].to_numpy()
sems = onsets.sem().loc[order].to_numpy()
x = np.array([0, 1])
color = [utils.colors[x] for x in order]
ax.barh(x, means, color=color, **bar_kwargs)
ax.errorbar(y=x, x=means, xerr=sems, **errorbar_kwargs)
ax.set_yticks(x)
ax.set_yticklabels([x.replace(" ", "\n") for x in order])
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Temporal onset within the dream")
ax.set_xlim(25, 50)
ax.xaxis.set(
    major_locator=plt.MultipleLocator(25), minor_locator=plt.MultipleLocator(5)
)
ax.set_ylim(-0.7, 1.7)
# ax.yaxis.set(major_locator=plt.MultipleLocator(0.1), minor_locator=plt.MultipleLocator(0.02))

stats = pg.wilcoxon(
    onsets["lucidity"], onsets["flying"], correction=True, method="approx"
)
p = stats.at["Wilcoxon", "p-val"]
stars = "*" * sum(p < cutoff for cutoff in [0.05, 0.01, 0.001])
stars_x = np.max(means + sems)
stars_x += 0.2
stars_color = "black" if stars_x else "gainsboro"
text_kwargs = dict(
    color=stars_color, fontsize=16, ha="center", va="center", rotation=270
)
ax.text(0.95, 0.5, stars, transform=ax.transAxes, **text_kwargs)
lines = ax.plot(
    [0.95, 0.95],
    [0, 1],
    color=stars_color,
    linewidth=2,
    transform=ax.get_yaxis_transform(),
)
for l in lines:
    l.set_dash_capstyle("round")
# l = mlines.Line2D([stars_x, stars_x], [0, 1], color=stars_color)
# l.set_dash_capstyle("round")
# ax.add_lines(l)

export_path = utils.deriv_dir / f"data-{dataset}_task-{task}_bars.png"
plt.savefig(export_path)
plt.close()


################################################################################
# heatmap of lucid vs flying?
################################################################################

first_event = onsets.idxmin(axis=1)
lucid_first_onsets = onsets[first_event.eq("lucidity")].reset_index(drop=True)
flying_first_onsets = onsets[first_event.eq("flying")].reset_index(drop=True)

for event in ["flying", "lucidity"]:
    c = utils.colors[event]
    if event == "flying":
        mat = flying_first_onsets.to_numpy().T
    elif event == "lucidity":
        mat = lucid_first_onsets.to_numpy().T
    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
    x = np.array([0, 1])
    color = [utils.colors[x] for x in order]
    plot_kwargs = dict(lw=1, alpha=0.1)
    ax.plot(mat, x, color=c, **plot_kwargs)
    ax.set_yticks(x)
    ax.set_yticklabels([x.replace(" ", "\n") for x in order])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Temporal onset within the dream")
    ax.set_xlim(0, 100)
    if event == "flying":
        ax.text(
            0,
            1,
            f"{event}-first dreams",
            color=c,
            ha="left",
            va="bottom",
            fontsize=10,
            transform=ax.get_yaxis_transform(),
        )
    elif event == "lucidity":
        ax.text(
            1,
            1,
            f"{event}-first dreams",
            color=c,
            ha="right",
            va="bottom",
            fontsize=10,
            transform=ax.get_yaxis_transform(),
        )
    ax.xaxis.set(
        major_locator=plt.MultipleLocator(25), minor_locator=plt.MultipleLocator(5)
    )
    ax.set_ylim(-0.7, 1.7)
    export_path = utils.deriv_dir / f"data-{dataset}_task-{task}_bars_{event}.png"
    plt.savefig(export_path)
    plt.close()
