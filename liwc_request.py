"""Run LIWC from command line."""

import argparse
import os
import subprocess
from time import sleep

import utils


available_datasets = ["dreamviews", "flying", "sddb"]

dictionaries = {
    "22": "LIWC22",
    "behav": "behavioral-activation-dictionary.dicx",
    "bigtwo": "big-two-agency-communion-dictionary.dicx",
    "bodytype": "body-type-dictionary.dicx",
    "eprime": "english-prime-dictionary.dicx",
    "foresight": "foresight-lexicon.dicx",
    "imagination": "imagination-lexicon.dicx",
    "mind": "mind-perception-dictionary.dicx",
    "physio": "physiological-sensations-dictionary.dicx",
    "qualia": "qualia-dictionary.dicx",
    "self": "self-determinationself-talk-dictionary.dicx",
    "sleep": "sleep-dictionary.dicx",
    "threat": "threat-dictionary.dicx",
    "vestibular": "vestibular.dic",
    "weiref": "weighted-referential-activity-dictionary.dicx",
    "wellbeing": "well-being-dictionary.dicx",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset", required=True, type=str, choices=available_datasets
)
parser.add_argument(
    "-o",
    "--overwrite",
    action="store_true",
    help="Overwrite output file if it already exists.",
)
args = parser.parse_args()

dataset = args.dataset
overwrite = args.overwrite


skip_header = "yes"
n_segments = [1, 5]
precision = 6
threads = -1
exclude_categories = ",".join(
    [
        "WC",
        "Analytic",
        "Clout",
        "Authentic",
        "Tone",
        "WPS",
        "BigWords",
        "Dic",
        "AllPunc",
        "Period",
        "Comma",
        "QMark",
        "Exclam",
        "Apostro",
        "OtherP",
    ]
)
# Open the LIWC-22 desktop app.
p = subprocess.Popen("C:\\Program Files\\LIWC-22\\LIWC-22.exe")
sleep(10)  # Give it a few seconds to open up.

if dataset == "flying":
    df = utils.load_sourcedata(dreams_only=False)
elif dataset == "dreamviews":
    df = utils.load_dreamviews()
elif dataset == "sddb":
    df = utils.load_sddb()

temp_file_path = "./temp.csv"
df.to_csv(temp_file_path, index=True)

column_indices = 1 + df.reset_index().columns.tolist().index("dream_text")
row_id_indices = 1

for dictx_id, dictx_path in dictionaries.items():
    for nof in n_segments:
        export_path = (
            utils.deriv_dir / f"data-{dataset}_liwc-{dictx_id}_nsegs-{nof}.csv"
        )
        if not export_path.exists() or overwrite:
            command = (
                "LIWC-22-cli --mode wc"
                f" --dictionary {dictx_path}"
                f" --input {temp_file_path}"
                f" --output {export_path}"
                f" --precision {precision}"
                f" --threads {threads}"
                f" --segmentation nof={nof}"
                f" --column-indices {column_indices}"
                f" --row-id-indices {row_id_indices}"
                f" --skip-header {skip_header}"
            )
            if dictx_id != "22":
                full_dict_path = str(utils.source_dir / "dictionaries" / dictx_path)
                command = command.replace(dictx_path, full_dict_path)
                command += f" --exclude-categories {exclude_categories}"
            # Run shell command and exit upon failure.
            subprocess.call(command.split())
            # result = subprocess.run(command, shell=True)
            # if result.returncode != 0:
            #     sys.exit()

p.terminate()
os.remove(temp_file_path)
