"""Turn every training shoe photo into a CLIP 'embedding': 512 numbers that describe what the image shows.

CLIP was trained to put an image and its caption close together, so it also embeds TEXT:
  image of a Nike Air Force 1  -> vector close to text "a photo of a Nike sneaker"
That lets us ask about brands we have NO training images of (Puma, Reebok, Vans, Asics).
Output: runs/clip_<model>.npz with image_embeddings (735, 512), flipped_embeddings (735, 512),
        text_embeddings (7, 512) for the 7 brand prompts, labels (735,)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image, ImageOps

TASK_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = TASK_FOLDER / "data" / "task_vietnam"
RUNS_FOLDER = TASK_FOLDER / "runs"
ALL_BRANDS = ["nike", "adidas", "converse", "puma", "reebok", "vans", "asics"]
torch.set_num_threads(4)

model_name, pretrained_tag = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("ViT-B-32", "openai")
clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
clip_model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

labels_table = pd.read_csv(DATA_FOLDER / "train-output.csv")


@torch.no_grad()
def embed_images(pil_images):
    pixel_batch = torch.stack([preprocess(image) for image in pil_images])
    embeddings = clip_model.encode_image(pixel_batch)
    return torch.nn.functional.normalize(embeddings, dim=-1).numpy()   # length 1 -> dot product = cosine similarity


image_embeddings, flipped_embeddings = [], []
for start in range(0, len(labels_table), 32):
    batch_images = [Image.open(DATA_FOLDER / "train-input" / name).convert("RGB") for name in labels_table.filename[start:start + 32]]
    image_embeddings.append(embed_images(batch_images))
    flipped_embeddings.append(embed_images([ImageOps.mirror(image) for image in batch_images]))

with torch.no_grad():
    prompts = [f"a photo of a {brand} sneaker" for brand in ALL_BRANDS]
    text_embeddings = torch.nn.functional.normalize(clip_model.encode_text(tokenizer(prompts)), dim=-1).numpy()

output_path = RUNS_FOLDER / f"clip_{model_name}_{pretrained_tag}.npz"
np.savez(output_path, image_embeddings=np.concatenate(image_embeddings), flipped_embeddings=np.concatenate(flipped_embeddings),
         text_embeddings=text_embeddings, labels=labels_table.label.to_numpy(), filenames=labels_table.filename.to_numpy())
print("saved", output_path, np.concatenate(image_embeddings).shape)
