"""Visualize the dataset with descriptive plots."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import utils

utils.load_matplotlib_settings()

# Load data.
df = utils.load_sourcedata(dreams_only=False).drop(columns="GPT_ID500")
assert df[["source_id", "subject_id", "report_type", "dream_text"]].notna().all().all()


################################################################################
# Visualize number of dreams vs comments for each source.
################################################################################

order = ["dream", "comment"]
counts = df.groupby("source_id")["report_type"].value_counts(dropna=False).unstack(fill_value=0)
counts = counts.reindex(counts.sum(axis=1).sort_values(ascending=False).index)
counts = counts[order]

counts2 = counts.rename_axis(columns=None)
counts2.loc["total"] = counts2.sum()
export_path = utils.deriv_dir / "data-flying_sample-type.csv"
counts2.to_csv(export_path, index=True)

fig, ax = plt.subplots(figsize=(3, 3.5), constrained_layout=True)
bar_kwargs = dict(width=0.8, linewidth=1, edgecolor="black")
palette = utils.colors.copy()

x = np.arange(len(counts))
ybottom = np.zeros(x.size)
for col in counts:
    y = counts[col].values
    c = palette[col]
    ax.bar(x, y, bottom=ybottom, color=c, **bar_kwargs)
    ybottom += y

ax.set_xticks(x)
ax.set_xticklabels(counts.index.values, rotation=40, ha="right", fontsize=8)

ax.set_xlabel("Data source")
ax.set_ylabel("Number of reports")

ax.set_ybound(upper=3200)
ax.yaxis.set(major_locator=plt.MultipleLocator(500), minor_locator=plt.MultipleLocator(100))
# ax.set_yscale("log")

handles = [ 
    plt.matplotlib.patches.Patch(
        facecolor=palette[column], label=column, edgecolor="black", linewidth=0.5
    )
    for column in counts
]

legend = ax.legend(
    handles=handles,
    loc="upper right",
    title="Report type",
    frameon=False,
    title_fontsize=8,
    fontsize=8,
    handletextpad=0.4,  # space between legend marker and label
    labelspacing=0.1,  # vertical space between the legend entries
    borderaxespad=0,
)

legend._legend_box.align = "left"
# legend._legend_box.sep = 5 # brings title up farther on top of handles/labels

export_path = utils.deriv_dir / "data-flying_sample-type.png"
plt.savefig(export_path)
plt.close()



################################################################################
# Visualize lucid dream frequency
################################################################################


# responses = {}
# completions = utils.load_json(utils.deriv_dir / f"data-flying_task-islucid_responses.json")
# for dream_id, completion in completions.items():
#     responses[dream_id] = [choice["message"]["content"] for choice in completion["choices"]]
# ser = (
#     pd.Series(responses).explode()
#     .rename("lucidity").rename_axis("dream_ID")
#     .map({"True": "lucid", "False": "non-lucid"})
# )
ser = utils.load_gpt_lucidity_codes(dataset="flying")
drms = df.join(ser, how="inner")


order = ["non-lucid", "lucid"]
counts = drms.groupby("source_id")["lucidity"].value_counts(dropna=False).unstack(fill_value=0)
counts = counts.reindex(counts.sum(axis=1).sort_values(ascending=False).index)
counts = counts[order]

counts2 = counts.rename_axis(columns=None)
counts2.loc["total"] = counts2.sum()
export_path = utils.deriv_dir / "data-flying_sample-lucidity.csv"
counts2.to_csv(export_path, index=True)

fig, ax = plt.subplots(figsize=(3, 3.5), constrained_layout=True)
bar_kwargs = dict(width=0.8, linewidth=1, edgecolor="black")
palette = utils.colors.copy()

x = np.arange(len(counts))
ybottom = np.zeros(x.size)
for col in counts:
    y = counts[col].values
    c = palette[col]
    ax.bar(x, y, bottom=ybottom, color=c, **bar_kwargs)
    ybottom += y

ax.set_xticks(x)
ax.set_xticklabels(counts.index.values, rotation=40, ha="right", fontsize=8)

ax.set_xlabel("Data source")
ax.set_ylabel("Number of dreams")

ax.set_ybound(upper=2200)
ax.yaxis.set(major_locator=plt.MultipleLocator(500), minor_locator=plt.MultipleLocator(100))
# ax.set_yscale("log")

handles = [ 
    plt.matplotlib.patches.Patch(
        facecolor=palette[column], label=column, edgecolor="black", linewidth=0.5
    )
    for column in counts
]

legend = ax.legend(
    handles=handles,
    loc="upper right",
    title="Lucidity",
    frameon=False,
    title_fontsize=8,
    fontsize=8,
    handletextpad=0.4,  # space between legend marker and label
    labelspacing=0.1,  # vertical space between the legend entries
    borderaxespad=0,
)

legend._legend_box.align = "left"
# legend._legend_box.sep = 5 # brings title up farther on top of handles/labels

export_path = utils.deriv_dir / "data-flying_sample-lucidity.png"
plt.savefig(export_path)
plt.close()



################################################################################
# Visualize number of authors for DREAMS vs gender count for each source.
################################################################################

dreams = df.query("report_type=='dream'")
dream_subjects = dreams.drop_duplicates("subject_id")

order = ["female", "male", "they", "unspecified"]
counts = dream_subjects.groupby("source_id")["sex"].value_counts(dropna=False).unstack(fill_value=0)
counts = counts.reindex(counts.sum(axis=1).sort_values(ascending=False).index)
counts = counts[order]

counts2 = counts.rename_axis(columns=None)
counts2.loc["total"] = counts2.sum()
export_path = utils.deriv_dir / "data-flying_sample-sex.csv"
counts2.to_csv(export_path, index=True)

fig, ax = plt.subplots(figsize=(3, 3.5), constrained_layout=True)
bar_kwargs = dict(width=0.8, linewidth=0, edgecolor="black")
palette = utils.colors.copy()

x = np.arange(len(counts))
ybottom = np.zeros(x.size)
for col in counts:
    y = counts[col].values
    c = palette[col]
    ax.bar(x, y, bottom=ybottom, color=c, **bar_kwargs)
    ybottom += y

ax.set_xticks(x)
ax.set_xticklabels(counts.index.values, rotation=40, ha="right", fontsize=8)

ax.set_xlabel("Data source")
ax.set_ylabel("Number of participants")

# ax.set_ybound(upper=3200)
# ax.yaxis.set(major_locator=plt.MultipleLocator(500), minor_locator=plt.MultipleLocator(100))

handles = [ 
    plt.matplotlib.patches.Patch(
        facecolor=palette[column], label=column, edgecolor="black", linewidth=0.5
    )
    for column in counts
]

legend = ax.legend(
    handles=handles,
    loc="upper right",
    title="Sex",
    frameon=False,
    title_fontsize=8,
    fontsize=8,
    handletextpad=0.4,  # space between legend marker and label
    labelspacing=0.1,  # vertical space between the legend entries
    borderaxespad=0,
)

legend._legend_box.align = "left"
# legend._legend_box.sep = 5 # brings title up farther on top of handles/labels

export_path = utils.deriv_dir / "data-flying_sample-sex.png"
plt.savefig(export_path)
plt.close()


################################################################################
# Visualize relationship between dream/author sample size for each source.
################################################################################

counts = dreams.groupby("source_id")["subject_id"].agg(["nunique", "count"])

counts2 = counts.rename_axis(columns=None)
counts2.loc["total"] = counts2.sum()
export_path = utils.deriv_dir / "data-flying_sample-authors.csv"
counts2.to_csv(export_path, index=True)

markers = {
    "DreamBank": "o",
    "LD4all": "s",
    "Misc": "+",
    "Reddit": "^",
    "SDDb": "*",
}

fig, ax = plt.subplots(figsize=(2.5, 2.5), constrained_layout=True)
scatter_kwargs = dict(s=60, c=utils.colors["dream"], clip_on=False)

for idx, row in counts.iterrows():
    m = markers[idx]
    ax.scatter(row["count"], y=row["nunique"], marker=m, **scatter_kwargs)

ax.plot([0, 1], [0, 1], "--", lw=1, color="black", zorder=1, transform=ax.transAxes)

ax.set_xlabel("Number of dreams")
ax.set_ylabel("Number of unique authors")

ax.set_xlim(0, 3200)
ax.set_ylim(0, 3200)
ax.yaxis.set(major_locator=plt.MultipleLocator(1000), minor_locator=plt.MultipleLocator(200))
ax.xaxis.set(major_locator=plt.MultipleLocator(1000), minor_locator=plt.MultipleLocator(200))
ax.set_aspect(1)

handles = [
    Line2D(
        [0], [0], marker=markers[x],
        color=utils.colors["dream"], linewidth=0, label=x,
        markerfacecolor=utils.colors["dream"], markersize=6
    )
    for x in counts.index
]

legend = ax.legend(
    handles=handles,
    loc="upper left",
    title="Data source",
    frameon=False,
    title_fontsize=8,
    fontsize=8,
    handletextpad=0.4,  # space between legend marker and label
    labelspacing=0.1,  # vertical space between the legend entries
    borderaxespad=0,
)

legend._legend_box.align = "left"
# legend._legend_box.sep = 5 # brings title up farther on top of handles/labels

export_path = utils.deriv_dir / "data-flying_sample-authors.png"
plt.savefig(export_path)
plt.close()



################################################################################
# Histogram zoomed in on LD4all dreams to view dreams per author
################################################################################

counts = dreams.query("source_id=='LD4all'")["subject_id"].value_counts()

fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
hist_kwargs = dict(bins=30, color=utils.colors["dream"], edgecolor="black", linewidth=1)

ax.hist(counts.to_numpy(), **hist_kwargs)

ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Number of dreams")
ax.set_ylabel("Number of dreamers")

ax.set_yscale("log")
ax.set_ybound(upper=1000)
ax.xaxis.set(major_locator=plt.MultipleLocator(100), minor_locator=plt.MultipleLocator(20))

ax.text(1, 1, "LD4all sample", ha="right", va="top", fontsize=10, transform=ax.transAxes)

export_path = utils.deriv_dir / "data-flying_sample-ld4all.png"
plt.savefig(export_path)
plt.close()
