import argparse
import os
import uuid

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import numpy as np
import torch
from fiftyone import ViewField as F
from PIL import Image
from rich import print
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def fetch(folder:str=None) -> dict:
    if os.path.exists(f'{os.getenv("HOME")}/.cache/diex/{folder.split("/")[-1]}/embeddings.pth'):
        return torch.load(f'{os.getenv("HOME")}/.cache/diex/{folder.split("/")[-1]}/embeddings.pth')
    else:
        return None

def extract_embeddings(path:str=None, processor:AutoImageProcessor=None, model:AutoModel=None, device:str='cuda') -> torch.Tensor:
    """
    Extracts DINO embeddings for a given image.
    Args:
        path (str): Path to the image file.

    Returns:
        torch.Tensor: The extracted embeddings.
    """
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.detach().cpu().numpy().flatten()

def main():
    parser = argparse.ArgumentParser(prog='DINO Explorer', description='A tool to explore DINO embeddings')
    parser.add_argument('f', nargs='+', type=str, help='List of folders to visualize, separated by space')
    parser.add_argument('--m', type=str, choices=['umap', 'tsne', 'pca'], default='umap', help='Method to compute visualization. Default: umap. Options: umap, tsne, pca')
    parser.add_argument('--d', type=int, default=0, help='GPU device number')
    parser.add_argument('--p', type=int, help='Port number to launch the FiftyOne app')
    parser.add_argument('--force', action='store_true', help='Force recompute embeddings')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.d}')
        print(f'[magenta]Using GPU {args.d}[/magenta]')
    else:
        device = torch.device('cpu')
        print(f'[bright_red]GPU not found; Using CPU instead![/bright_red]')

    dataset = fo.Dataset(str(uuid.uuid4()))
    final_embeds = []
    embeds = dict()
    
    for folder in args.f:
        precomputed_embeds = fetch(folder)
        if precomputed_embeds is None or args.force:
            print(f'[cyan]Generating embeddings for [bold]{folder.split("/")[-1]}[/bold][/cyan]')
            os.makedirs(f'{os.getenv("HOME")}/.cache/diex/{folder.split("/")[-1]}', exist_ok=True)
            processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
            model = AutoModel.from_pretrained('facebook/dinov2-giant').to(device)       
            files = os.listdir(folder)
            for file in tqdm(files):
                embeds[file] = extract_embeddings(path=os.path.join(folder, file), processor=processor, model=model, device=device)
                final_embeds.append(embeds[file])
                sample = fo.Sample(filepath=os.path.join(folder,file), tags=[f'{folder.split("/")[-1]}'])
                dataset.add_sample(sample)
            torch.save(embeds, f'{os.getenv("HOME")}/.cache/diex/{folder.split("/")[-1]}/embeddings.pth')
        else:
            print(f'[green]Using cached embeddings for [bold]{folder.split("/")[-1]}[/bold][/green]')
            files = os.listdir(folder)
            for file in tqdm(files):
                sample = fo.Sample(filepath=os.path.join(folder,file), tags=[f'{folder.split("/")[-1]}'])
                dataset.add_sample(sample)
                final_embeds.append(precomputed_embeds[file])

    final_embeds = np.stack(final_embeds)
    results = fob.compute_visualization(
        dataset,
       embeddings=final_embeds,
       brain_key ='img_viz',
       method=args.m,
       verbose=False
    )

    if args.p is None:
        session = fo.launch_app(dataset)
    else:
        session = fo.launch_app(dataset, port=args.p)
    session.wait(0)

if __name__ == '__main__':
    main()
