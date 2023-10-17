# flying

A study of flying dreams.


## ChatGPT coding

```shell
# Is a post a dream?
python gpt_request.py --dataset flying --task isdream       #> data-flying_task-isdream_responses.json

# Is a dream lucid?
python gpt_request.py --dataset dreamviews --task islucid   #> data-dreamviews_task-islucid_responses.json
python gpt_request.py --dataset flying --task islucid       #> data-flying_task-islucid_responses.json
python gpt_request.py --dataset sddb --task islucid         #> data-sddb_task-islucid_responses.json

# Identify themes in a dream
python gpt_request.py --dataset flying --task thematicT     #> data-flying_task-thematicT_responses.json

# Annotate non-dream, lucid dream, and flying dream sections
python gpt_request.py --dataset flying --task annotate      #> data-flying_task-annotate_responses.json
```

## Visualizations

```shell
# Describe the sample size and demographics of the dataset
python plot_descriptives.py         #> data-flying_sample-*.png

# Plot top technique themes for lucid and non-lucid dreams (separately)
python plot_themes_lucidity.py      #> data-flying_themes-techniq_lucidity.png

# Plot timecourses based on GPT supp/flying/lucid annotations (and a bar graph)
python plot_timecourses.py          #> data-flying_task-annotate_*.png
```
