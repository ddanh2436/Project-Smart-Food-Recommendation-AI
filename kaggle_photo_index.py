# -*- coding: utf-8 -*-
"""Check every restaurant's cover photo with SigLIP2-gopt, on a Kaggle GPU.

What it writes, per restaurant, into MongoDB:

    photo: {
      kind: "food" | "logo" | "menu" | "banner" | "storefront" | "people",
      dishGuess: [{tag: "Phở", score: 0.91}, ...],   # top 3, tags as stored
      matchesTags: true,       # top guess is one of the restaurant's own tags
      url: <the avatarUrl that was checked>,
      model: "ViT-gopt-16-SigLIP2-384",
      checkedAt: <date>
    }

Why offline: the model has about 1.9 billion parameters and runs in seconds
on a T4 but not at all on the free CPU Space. Everything it can tell us about
the stored photos is computed once here; the Space only reads the result.

Kaggle setup: Accelerator = GPU T4 (x1 is enough), Internet = On, and a Secret
named MONGO_URI (Add-ons > Secrets). Run the cells in order. Cell 5 writes in
batches and skips photos already checked, so if the session stops, run cells
1 -> 5 again and it carries on.
"""

# %% Cell 1 - install and fetch the code
# open_clip >= 3.0 knows the SigLIP2 models; pymongo for the database.
# !pip install -q "open_clip_torch>=3.0" "pymongo[srv]"
# !rm -rf /kaggle/working/ai
# !git clone -q --depth 1 -b refactor/security-fixes-and-ai-upgrade \
#     https://github.com/ddanh2436/Project-Smart-Food-Recommendation-AI.git /kaggle/working/ai

# %% Cell 2 - settings and secrets
import os
import sys

import torch
from kaggle_secrets import UserSecretsClient

os.environ["MONGO_URI"] = UserSecretsClient().get_secret("MONGO_URI")
sys.path.insert(0, "/kaggle/working/ai")

MODEL = "ViT-gopt-16-SigLIP2-384"
HF_ID = f"hf-hub:timm/{MODEL}"
DB_NAME = "VietNomNom"
BATCH = 32          # images per forward pass; lower to 16 if CUDA runs out of memory
FLUSH_EVERY = 200   # restaurants per database write

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "- no GPU, turn it on")

# %% Cell 3 - model and label texts
import open_clip
import torch.nn.functional as F

from dish_labels import DISH_PROMPTS, NON_FOOD_PROMPTS

model, preprocess = open_clip.create_model_from_pretrained(HF_ID)
tokenizer = open_clip.get_tokenizer(HF_ID)
model = model.to(device).half().eval()

dish_tags = list(DISH_PROMPTS)
non_food = list(NON_FOOD_PROMPTS)
texts = (
    [f"a photo of {DISH_PROMPTS[t]}, Vietnamese food" for t in dish_tags]
    + [NON_FOOD_PROMPTS[k] for k in non_food]
)
with torch.no_grad():
    tokens = tokenizer(texts, context_length=model.context_length).to(device)
    text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
scale = model.logit_scale.exp().float()
bias = model.logit_bias.float() if getattr(model, "logit_bias", None) is not None else 0.0
print(len(dish_tags), "dishes +", len(non_food), "non-food labels")

# %% Cell 4 - which restaurants still need a check
from datetime import datetime, timezone

from pymongo import MongoClient, UpdateOne

client = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=20_000)
restaurants = client[DB_NAME]["restaurants"]

todo = []
for doc in restaurants.find({}, {"avatarUrl": 1, "tags": 1, "photo": 1}):
    url = doc.get("avatarUrl") or ""
    done = doc.get("photo") or {}
    if url.startswith("http") and not (done.get("model") == MODEL and done.get("url") == url):
        todo.append(doc)
print(f"{len(todo)} photos to check")

# %% Cell 5 - download, encode, classify, write
import io
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (VietNomNom photo check)"


def fetch(doc):
    try:
        response = session.get(doc["avatarUrl"], timeout=15)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return doc, preprocess(image)
    except Exception:
        return doc, None


def own_tags(doc):
    tags = doc.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip(" '\"") for t in tags.strip("[]").split(",")]
    return {str(t).strip().lower() for t in tags}


def classify(image_batch):
    with torch.no_grad():
        features = F.normalize(model.encode_image(image_batch.to(device).half()).float(), dim=-1)
        # SigLIP scores each label independently (sigmoid), so these are
        # per-label probabilities, not a distribution that sums to one.
        return torch.sigmoid(features @ text_features.T * scale + bias).cpu()


writes, checked, food, started = [], 0, 0, time.time()
n_dish = len(dish_tags)
with ThreadPoolExecutor(max_workers=16) as pool:
    for start in range(0, len(todo), BATCH):
        loaded = [(d, t) for d, t in pool.map(fetch, todo[start:start + BATCH]) if t is not None]
        if not loaded:
            continue
        probs = classify(torch.stack([t for _, t in loaded]))
        for (doc, _), p in zip(loaded, probs):
            dish_p, other_p = p[:n_dish], p[n_dish:]
            best_other = int(other_p.argmax())
            is_food = float(dish_p.max()) >= float(other_p[best_other])
            top = torch.topk(dish_p, 3)
            guess = [
                {"tag": dish_tags[int(i)], "score": round(float(s), 4)}
                for s, i in zip(top.values, top.indices)
            ]
            kind = "food" if is_food else non_food[best_other]
            food += is_food
            writes.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"photo": {
                "kind": kind,
                "dishGuess": guess,
                "matchesTags": guess[0]["tag"].lower() in own_tags(doc),
                "url": doc["avatarUrl"],
                "model": MODEL,
                "checkedAt": datetime.now(timezone.utc),
            }}}))
            checked += 1
        if len(writes) >= FLUSH_EVERY:
            restaurants.bulk_write(writes, ordered=False)
            writes.clear()
            rate = checked / (time.time() - started)
            print(f"{checked}/{len(todo)}  food {food}  {rate:.1f}/s", flush=True)
if writes:
    restaurants.bulk_write(writes, ordered=False)
print(f"Done: {checked} checked, {food} food photos, {time.time() - started:.0f}s")

# %% Cell 6 - summary (optional)
from collections import Counter

rows = list(restaurants.find({"photo.model": MODEL}, {"photo": 1, "tenQuan": 1}))
print(Counter(r["photo"]["kind"] for r in rows))
print("top guess matches own tag:", sum(r["photo"]["matchesTags"] for r in rows), "/", len(rows))
for r in [r for r in rows if r["photo"]["kind"] != "food"][:8]:
    print(r["photo"]["kind"], "|", r["tenQuan"][:40])
