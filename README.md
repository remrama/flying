# flying

A study of the phenomenology of flying dreams.


## Ask ChatGPT to code and annotate dreams

```shell
# Is it a dream?
python run_gpt_task.py --task isdream   # --> task-isdream_responses.json
# Is a dream lucid?
python run_gpt_task.py --task islucid   # --> task-islucid_responses.json
# Identify themes in a dream
python run_gpt_task.py --task themesT   # --> task-themesT_responses.json
python run_gpt_task.py --task themesD   # --> task-themesD_responses.json
python run_gpt_task.py --task themesM   # --> task-themesM_responses.json
# Annotate non-dream, lucid dream, and flying dream sections
python run_gpt_task.py --task annotate  # --> task-annotate_responses.json
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
