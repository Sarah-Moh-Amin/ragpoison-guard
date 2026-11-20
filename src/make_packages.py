# make_packages.py
# Build the two submission ZIPs in the project folder:
#   Sarah_RAGPoison_Guard_Task_1.zip and Sarah_RAGPoison_Guard_Task_2.zip
# It stops with a clear message if a required file is missing.
# It never adds .venv, .git, model downloads or anything from ~/.cache.
# Usage: python src/make_packages.py

import glob
import os
import sys
import zipfile

FINAL_RUN = "runs/baseline_20261120_210939"
TASK1_NAME = "Sarah_RAGPoison_Guard_Task_1"
TASK2_NAME = "Sarah_RAGPoison_Guard_Task_2"


def stop(message):
    # Print a clear message and stop the program.
    print("CHECK FAILED:", message)
    sys.exit(1)


def files_in_folder(folder):
    # All files under a folder (sorted), skipping hidden files like .DS_Store.
    found = []
    for root, dirs, names in os.walk(folder):
        dirs.sort()
        for name in sorted(names):
            if not name.startswith("."):
                found.append(os.path.join(root, name))
    return found


# Step 1: list every (source file, path inside the ZIP) for Task 1.
task1 = [
    ("reports/task1_tutorial.docx", TASK1_NAME + "/report/task1_tutorial.docx"),
    ("reports/task1_presentation.pptx", TASK1_NAME + "/presentation/task1_presentation.pptx"),
]

# Step 2: the same for Task 2.
task2 = [
    ("README.md", TASK2_NAME + "/README.md"),
    ("reports/task2_report.docx", TASK2_NAME + "/report/task2_report.docx"),
]
# source/: all Python files, config and the package lock file.
for path in sorted(glob.glob("src/*.py")):
    task2.append((path, TASK2_NAME + "/source/" + path))
task2.append(("config.json", TASK2_NAME + "/source/config.json"))
task2.append(("requirements-lock.txt", TASK2_NAME + "/source/requirements-lock.txt"))
# data/: clean data, manifests and the data card.
for path in files_in_folder("data/clean") + files_in_folder("data/manifests"):
    task2.append((path, TASK2_NAME + "/" + path))
task2.append(("data/DATA_CARD.md", TASK2_NAME + "/data/DATA_CARD.md"))
# results/: the whole runs/ folder and resource_notes.txt.
for path in files_in_folder("runs"):
    task2.append((path, TASK2_NAME + "/results/" + path))
task2.append(("resource_notes.txt", TASK2_NAME + "/results/resource_notes.txt"))
# results/: copies of the final v3 files at the top of results/.
for name in ["baseline_results.jsonl", "baseline_summary.csv", "final_metrics.csv",
             "timing.csv", "trace_Q001.json"]:
    task2.append((FINAL_RUN + "/" + name, TASK2_NAME + "/results/" + name))
# results/embedding_cache/: the embedding cache with its metadata.
for name in ["passage_embeddings.npy", "passage_ids.json", "cache_metadata.json"]:
    task2.append(("cache/" + name, TASK2_NAME + "/results/embedding_cache/" + name))

# Step 3: check that every file exists before building anything.
missing = [src for src, _ in task1 + task2 if not os.path.isfile(src)]
if missing:
    stop("missing required files:\n  " + "\n  ".join(missing))
if not glob.glob("src/*.py"):
    stop("no Python files found in src/")

# Step 4: safety check, nothing from .venv, .git or a model/download cache.
for src, _ in task1 + task2:
    parts = src.split(os.sep)
    if ".venv" in parts or ".git" in parts or ".cache" in parts or "__pycache__" in parts:
        stop("refusing to package " + src)


def build_zip(zip_name, entries):
    # Write the ZIP, then print its file list and size.
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arcname in entries:
            z.write(src, arcname)
    print("\n" + zip_name)
    for _, arcname in entries:
        print("  ", arcname)
    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print("  ", len(entries), "files,", round(size_mb, 2), "MB")


# Step 5: build both ZIPs.
build_zip(TASK1_NAME + ".zip", task1)
build_zip(TASK2_NAME + ".zip", task2)
print("\nDone.")
