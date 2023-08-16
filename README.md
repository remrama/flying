# flying

A study of flying dreams.


## Fine-tuning an OpenAI model to preprocess dream reports

```shell
# Read in and export OpenAI API key.
export OPENAI_API_KEY=$(head -n 1 .openai)

# Convert the dataset of training examples to jsonl format.
python csv2jsonl.py

# Let OpenAI run their quality control on the dataset.
openai tools fine_tunes.prepare_data --file dreams.jsonl --quiet

# Create the fine-tuned model.
openai api fine_tunes.create \
    --training_file dreams.jsonl \
    --suffix flying \
    --model davinci \
    --n_epochs 2

# Check on model with these commands
#   openai api fine_tunes.list
#   openai api fine_tunes.get --id FINE_TUNED_JOB_ID
#   openai api fine_tunes.cancel --id FINE_TUNED_JOB_ID
#   openai api models.delete -id FINE_TUNED_JOB_ID

# # Download results file.
# openai api fine_tunes.results --id FINE_TUNED_JOB

# # Test it!
# openai api completions.create -m FINE_TUNED_MODEL -p YOUR_PROMPT
```
