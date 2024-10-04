"""Explore LIWC results."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import utils


# Load custom matplotlib settings
utils.load_matplotlib_settings()

# Define dataset and number of segments
dataset = "flying"
n_segments = 1

# List of LIWC dictionaries to use
liwc_dicts = [
    "22",
    "bigtwo",
    "vestibular",
]

# Load and preprocess the flying dataset
flying = utils.load_sourcedata(dreams_only=True).drop(columns="GPT_ID500")
flying_lucid_ser = utils.load_gpt_lucidity_codes(dataset="flying")
flying = flying.join(flying_lucid_ser, how="inner")

# Load additional datasets
dreamviews = utils.load_dreamviews()
sddb = utils.load_sddb()

# Load LIWC results
def load_liwc_file(dataset, dic, nsegs=n_segments):
    import_name = f"data-{dataset}_liwc-{dic}_nsegs-{nsegs}.csv"
    import_path = utils.deriv_dir / import_name
    return (
        pd.read_csv(import_path)
        .rename(columns={"Row ID": "dream_id"})
        .set_index(["dream_id", "Segment"])
        # .drop(columns="dream")
    )

# Load LIWC data for the flying dataset using all specified dictionaries
flying_liwc = pd.concat([load_liwc_file("flying", d) for d in liwc_dicts], axis=1)
# Load LIWC data for the DreamViews dataset using all specified dictionaries
dreamviews_liwc = pd.concat(
    [load_liwc_file("dreamviews", d) for d in liwc_dicts], axis=1
)
# Load LIWC data for the SDDb dataset using all specified dictionaries
sddb_liwc = pd.concat([load_liwc_file("sddb", d) for d in liwc_dicts], axis=1)

# List of vestibular-related categories
vestibular_cats = [
    "ascend",
    "balance",
    "descend",
    "dizzy",
    "falling",
    "float",
    "fly",
    "gyrate",
    "hover",
    "soar",
    "spin",
    "spinning",
    "swirl",
    "vertigo",
    "weightless",
    "whirl",
]

# Calculate the sum of vestibular-related categories for each dataset
flying_liwc["vestib"] = flying_liwc[vestibular_cats].sum(axis=1)
dreamviews_liwc["vestib"] = dreamviews_liwc[vestibular_cats].sum(axis=1)
sddb_liwc["vestib"] = sddb_liwc[vestibular_cats].sum(axis=1)

# Join lucidity codes to the LIWC data
flying_liwc = flying_liwc.join(flying["lucidity"], how="inner")
dreamviews_liwc = dreamviews_liwc.join(dreamviews["lucidity"], how="inner")

# Drop the "Segment" index if there is only one segment
if n_segments == 1:
    flying_liwc = flying_liwc.droplevel("Segment")
    dreamviews_liwc = dreamviews_liwc.droplevel("Segment")
    sddb_liwc = sddb_liwc.droplevel("Segment")

# Melt the dataframes for easier plotting
fly_melt = flying_liwc.melt(
    id_vars="lucidity", var_name="category", value_name="frequency", ignore_index=False
)
dv_melt = dreamviews_liwc.melt(
    id_vars="lucidity", var_name="category", value_name="frequency", ignore_index=False
)
sddb_melt = sddb_liwc.melt(
    var_name="category", value_name="frequency", ignore_index=False
)

# Add a "dataset" column to each dataframe
fly_melt["dataset"] = "Flying"
dv_melt["dataset"] = "DreamViews"
sddb_melt["dataset"] = "SDDb"

# Combine the dataframes
flying_and_dv = pd.concat([fly_melt, dv_melt], axis=0)
flying_and_sddb = pd.concat([fly_melt.drop(columns="lucidity"), sddb_melt], axis=0)

# flying_sddb_dreams = df.query("source_id=='SDDb'")["dream"].tolist()
# sddb["flying"] = sddb["answer_text"].apply(
#     lambda x: x in flying_sddb_dreams
# )
# sddb = sddb[sddb["flying"] == False]
# sddb = sddb.drop(columns=["Row ID", "answer_text", "flying"])

################################################################################
# Plots
################################################################################

# Shows all individual vestibular categories within the Flying dataset
# sns.barplot(data=fly_melt, x="category", hue="lucidity", y="frequency", palette=utils.colors, order=vestibular_cats)
# Including dreamviews...
# sns.barplot(data=flying_and_dv[flying_and_dv["category"].isin(vestibular_cats)], x="category", hue="lucidity", y="frequency")

# Define categorical plotting orders
lucidity_order = ["non-lucid", "lucid"]
dataset_order_dv = ["Flying", "DreamViews"]
dataset_order_sddb = ["Flying", "SDDb"]
category = "Agency"

# Plot the frequency of category-related words in the Flying and DreamViews datasets
fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
sns.barplot(
    flying_and_dv.query(f"category=='{category}'"),
    x="dataset",
    hue="lucidity",
    y="frequency",
    saturation=1,
    palette=utils.colors,
    order=dataset_order_dv,
    hue_order=lucidity_order,
    ax=ax,
)

# Set plot labels
ax.set_xlabel("Dataset")
ax.set_ylabel(f"{category}-related word frequency".capitalize())
export_path = utils.deriv_dir / f"liwc-{category}_dreamviews.png"

# Save the plot
plt.savefig(export_path)
plt.close()

# Repeat the same process for other categories (copy/pasted garbage)
category = "insight"
fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
plot_df = flying_and_dv.query(f"category=='{category}'")
sns.barplot(
    data=plot_df,
    x="dataset",
    hue="lucidity",
    y="frequency",
    saturation=1,
    palette=utils.colors,
    order=dataset_order_dv,
    hue_order=lucidity_order,
    ax=ax,
)
ax.set_xlabel("Dataset")
ax.set_ylabel(f"{category}-related word frequency".capitalize())
export_path = utils.deriv_dir / f"liwc-{category}_dreamviews.png"
plt.savefig(export_path)
plt.close()

category = "emo_pos"
fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
sns.barplot(
    flying_and_dv.query(f"category=='{category}'"),
    x="dataset",
    hue="lucidity",
    y="frequency",
    saturation=1,
    palette=utils.colors,
    order=dataset_order_dv,
    hue_order=lucidity_order,
    ax=ax,
)
ax.set_xlabel("Dataset")
ax.set_ylabel("Positive emotion word frequency".capitalize())
export_path = utils.deriv_dir / f"liwc-{category}_dreamviews.png"
plt.savefig(export_path)
plt.close()

category = "emo_pos"
fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
sns.barplot(
    flying_and_dv.query(f"category=='{category}'"),
    x="dataset",
    hue="lucidity",
    y="frequency",
    saturation=1,
    palette=utils.colors,
    order=dataset_order_dv,
    hue_order=lucidity_order,
    ax=ax,
)
ax.set_xlabel("Dataset")
ax.set_ylabel(f"{category}-related word frequency".capitalize())
export_path = utils.deriv_dir / f"liwc-{category}_dreamviews.png"
plt.savefig(export_path)
plt.close()

# def plot(category):
#     category = "emo_neg"
#     nonflying = sddb[category].to_numpy()
#     # flying_nonlucid = drms.query("source_id=='SDDb'").query("lucidity=='non-lucid'")[category].to_numpy()
#     # flying_lucid = drms.query("source_id=='SDDb'").query("lucidity=='lucid'")[category].to_numpy()
#     flying_nonlucid = drms.query("lucidity=='non-lucid'")[category].to_numpy()
#     flying_lucid = drms.query("lucidity=='lucid'")[category].to_numpy()
#     fig, ax = plt.subplots(figsize=(2.5, 3), constrained_layout=True)
#     xvals = [0, 1]
#     yvals = [np.mean(flying_nonlucid), np.mean(flying_lucid)]
#     yerrs = [sem(flying_nonlucid), sem(flying_lucid)]
#     norm = np.mean(nonflying)
#     norm_sem = sem(nonflying)
#     colors = [palette["non-lucid"], palette["lucid"]]
#     ax.bar(x=xvals, height=yvals, yerr=yerrs, width=0.6, color=colors, edgecolor="black", lw=1)
#     ax.errorbar(x=xvals, y=yvals, yerr=yerrs, fmt="o", ms=3, c="black")
#     ax.set_xticks(xvals)
#     ax.axhline(norm, color="black", linewidth=1, linestyle="dashed")
#     ax.fill_between([0, 1], [norm-norm_sem]*2, [norm+norm_sem]*2, color="black", alpha=0.3, linewidth=0, transform=ax.get_yaxis_transform())
#     ax.text(1, norm+norm_sem, "Non-flying\ndream norm", color="black", fontsize=10, ha="left", va="center", transform=ax.get_yaxis_transform())
#     # ax.text(1, norm+norm_sem, "Non-flying dream norm", color="black", fontsize=10, ha="right", va="bottom", transform=ax.get_yaxis_transform())
#     ax.set_xticklabels(["Flying\nnon-lucid", "Flying\nlucid"])
#     ax.set_ylabel(f"{category} word frequency")
#     ax.spines[["top", "right"]].set_visible(False)

# def plot_bars(category):
#     category = "emo_neg"
#     nonflying = sddb[category].to_numpy()
#     # flying_nonlucid = drms.query("source_id=='SDDb'").query("lucidity=='non-lucid'")[category].to_numpy()
#     # flying_lucid = drms.query("source_id=='SDDb'").query("lucidity=='lucid'")[category].to_numpy()
#     flying_nonlucid = drms.query("lucidity=='non-lucid'")[category].to_numpy()
#     flying_lucid = drms.query("lucidity=='lucid'")[category].to_numpy()
#     fig, ax = plt.subplots(figsize=(2, 3), constrained_layout=True)
#     xvals = [0, 1, 2]
#     yvals = [np.mean(nonflying), np.mean(flying_nonlucid), np.mean(flying_lucid)]
#     yerrs = [sem(nonflying), sem(flying_nonlucid), sem(flying_lucid)]
#     colors = ["white", palette["non-lucid"], palette["lucid"]]
#     ax.bar(x=xvals, height=yvals, yerr=yerrs, color=colors, edgecolor="black", lw=1)
#     ax.errorbar(x=xvals, y=yvals, yerr=yerrs, c="black")
#     ax.set_xticks(xvals)
#     ax.set_xticklabels(["Non-flying", "Flying\nnon-lucid", "Flying\nlucid"])
#     ax.set_ylabel(f"{category} word frequency")


# ########## Validating lucid vs non-lucid dreams w/ agency and insight.

# for cat in ["insight", "Agency"]:
#     fig, ax = plt.subplots(figsize=(2, 3), constrained_layout=True)
#     sns.pointplot(
#         data=drms,
#         x="lucidity",
#         y=cat,
#         palette=palette,
#         order=["non-lucid", "lucid"],
#     )
#     if cat == "insight":
#         ax.set_ylim(2, 3)
#     elif cat == "Agency":
#         ax.set_ylim(3, 4)
#     ax.yaxis.set(major_locator=plt.MultipleLocator(0.2))

#     ax.set_xlabel("Lucidity")
#     ax.set_ylabel(f"{cat}-related word frequency".capitalize())
#     ax.spines[["top", "right"]].set_visible(False)

#     a, b = drms.groupby("lucidity")["insight"].apply(list)
#     stats = pg.mwu(a, b)

#     p = stats.at["MWU", "p-val"]
#     stars = "*" * sum(p < cutoff for cutoff in [0.05, 0.01, 0.001])
#     # stars_x = np.max(means + sems)
#     # stars_x += 0.2
#     stars_color = "black" if stars else "gainsboro"
#     text_kwargs = dict(color=stars_color, fontsize=16, ha="center", va="center")
#     ax.text(0.5, 0.95, stars, transform=ax.transAxes, **text_kwargs)
#     lines = ax.plot([0, 1], [0.95, 0.95], color=stars_color, linewidth=2, transform=ax.get_xaxis_transform())
#     for l in lines:
#         l.set_dash_capstyle("round")
#     # l = mlines.Line2D([stars_x, stars_x], [0, 1], color=stars_color)
#     # l.set_dash_capstyle("round")
#     # ax.add_lines(l)
#     ax.set_xlim(-0.5, 1.5)

#     export_path = utils.deriv_dir / f"lucidity_{cat}.png"
#     plt.savefig(export_path)
#     plt.close()


# ################################################################################
# # Language use BEFORE vs AFTER moment of flying?
# # A test of what triggered flying? in those where lucidity did not come first.
# ################################################################################

# """
# 1. Lucid dreams are positive and beneficial.
# 2. But lucid dreams are highly conflated with flying, as it's the #1 thing people do in their LDs.
# 3. Mechanism of LD benefits are unknown, but might be achievement, conflated with flying.
# 4. Our data show that yes, LD leads to flying. But we also have
#     - (a) flying leading to LD, and
#     - (b) non-lucid flying dreams.
# 5. So we can now ask
#     - (a) why does flying lead to LD?
#     - (b) are the non-lucid flying dreams also beneficial? bc if so, maybe lucidity isn't that great.
#         - do NLD flying dreams so achievement?
#         - increased positive emotion?
#         - Are they escaping fear?
#         - If any of the above, maybe we should just be inducing flying dreams.

# """
