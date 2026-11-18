# smoke_test.py
# Task 2, Day 1: check that the retriever (Contriever) and the
# generator (Qwen) both run on this computer, and record the hardware.

import platform
import shutil
import subprocess
import time

import numpy as np
import psutil
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


# ---------------------------------------------------------------
# Step 1: record the hardware in resource_notes.txt
# ---------------------------------------------------------------
print("Step 1: recording hardware...")

try:
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
except Exception:
    chip = platform.processor()

ram_gb = psutil.virtual_memory().total / (1024 ** 3)
free_disk_gb = shutil.disk_usage(".").free / (1024 ** 3)

notes = []
notes.append("Operating system: " + platform.platform())
notes.append("Python version: " + platform.python_version())
notes.append("Chip / CPU: " + chip)
notes.append("RAM (GB): " + str(round(ram_gb, 1)))
notes.append("Free disk (GB): " + str(round(free_disk_gb, 1)))
notes.append("GPU: Apple MPS available = " + str(torch.backends.mps.is_available()))
notes.append("PyTorch version: " + torch.__version__)
notes.append("Device used for this test: cpu")

for line in notes:
    print("   " + line)


# ---------------------------------------------------------------
# Step 2: test the retriever (Contriever)
# ---------------------------------------------------------------
print("\nStep 2: loading Contriever (first time downloads about 0.4 GB)...")

retriever_name = "facebook/contriever"
start = time.time()
tok = AutoTokenizer.from_pretrained(retriever_name)
encoder = AutoModel.from_pretrained(retriever_name).eval()
print("   loaded in", round(time.time() - start, 1), "seconds")


def encode(texts):
    # Turn a list of texts into vectors (one vector per text).
    inputs = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.inference_mode():
        hidden = encoder(**inputs).last_hidden_state
    # Mean pooling: average the word vectors, ignoring padding.
    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    return pooled.numpy()


passages = [
    "Aster Archive closes at 18:00 on weekdays.",
    "Aster Archive opens at 09:00 on weekdays.",
    "The city football club won the league last season.",
]
question = "When does Aster Archive close on weekdays?"

E = encode(passages)
q = encode([question])

print("   passage vectors shape:", E.shape)   # should be (3, 768)
print("   question vector shape:", q.shape)   # should be (1, 768)
print("   all numbers finite:", bool(np.isfinite(E).all()))

scores = E @ q[0]                              # dot product scores
print("   retrieval scores (higher = more relevant):")
for text, score in zip(passages, scores):
    print("     ", round(float(score), 3), "|", text)

retriever_revision = getattr(encoder.config, "_commit_hash", "unknown")
notes.append("Retriever: " + retriever_name + " revision " + str(retriever_revision))


# ---------------------------------------------------------------
# Step 3: test the generator (Qwen)
# ---------------------------------------------------------------
print("\nStep 3: loading Qwen (first time downloads about 3 GB, please wait)...")

generator_name = "Qwen/Qwen2.5-1.5B-Instruct"
start = time.time()
gtok = AutoTokenizer.from_pretrained(generator_name)
generator = AutoModelForCausalLM.from_pretrained(generator_name).eval()
load_time = time.time() - start
print("   loaded in", round(load_time, 1), "seconds")

prompt = (
    "Answer the question using only the evidence below.\n"
    "If the evidence is insufficient, answer ABSTAIN.\n\n"
    "Question: " + question + "\n\n"
    "Evidence:\n"
    "[P001] " + passages[0] + "\n"
    "[P002] " + passages[1] + "\n\n"
    "Answer:"
)

messages = [{"role": "user", "content": prompt}]
text = gtok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = gtok(text, return_tensors="pt")

start = time.time()
with torch.inference_mode():
    out = generator.generate(**inputs, max_new_tokens=64, do_sample=False,
                             pad_token_id=gtok.eos_token_id)
gen_time = time.time() - start

new_tokens = out[0, inputs["input_ids"].shape[1]:]   # keep only the new answer
answer = gtok.decode(new_tokens, skip_special_tokens=True)

print("   answer:", answer)
print("   generation time:", round(gen_time, 1), "seconds")

generator_revision = getattr(generator.config, "_commit_hash", "unknown")
notes.append("Generator: " + generator_name + " revision " + str(generator_revision))
notes.append("Generator load time (s): " + str(round(load_time, 1)))
notes.append("One answer generation time (s): " + str(round(gen_time, 1)))
notes.append("Smoke test answer: " + answer)


# ---------------------------------------------------------------
# Step 4: save everything to resource_notes.txt
# ---------------------------------------------------------------
with open("resource_notes.txt", "w") as f:
    f.write("\n".join(notes) + "\n")

print("\nDone. Results saved to resource_notes.txt")
