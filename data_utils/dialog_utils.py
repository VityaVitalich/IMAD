import os
from functools import partial
import json
from typing import Union, List, Dict, Any

from .image_utils import ImageDataset

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
from transformers import CLIPModel, CLIPTokenizer


class DialogDataset(Dataset):
    """A class containing a dialog dataset with optional scores.

    Attributes:
        image_scores (List[float]): CLIP generated scores.
        image_indices (List[int]): Indices of closest images in image dataset.
    """
    def __init__(
        self, dialogs: Union[List[str], str], min_length: int = 0, max_length: int = np.inf,
        path2features: str = None, model: CLIPModel = None, tokenizer: CLIPTokenizer = None,
        device: Union[torch.device, str] = 'cpu', indices: List[int] = None
    ):
        """
        Args:
            dialogs (Union[pd.Dataset, str]):
                Either a list of dialogs or a json file containing the dataset.
            min_length (int, optional): Min length of a sentence. Defaults to 0.
            max_length (int, optional): Max length of a sentence. Defaults to np.inf.
            path2features (str, optional):
                Path to directory containing saved features. Defaults to None.
            model (CLIPModel, optional):
                Huggingface CLIP model. Defaults to None.
            tokenizer (CLIPTokenizer, optional):
                Muggingface CLIP tokenizer. Defaults to None.
            device (Union[torch.device, str], optional):
                PyTorch device. Defaults to 'cpu'.
            indices (List[int], optional):
                Indices of dialog list or json to include in dataset. Defaults to None.
        """
        self.indices = indices
        if isinstance(dialogs, str) and dialogs.endswith('.json'):
            self.from_json(dialogs, indices=self.indices)
        else:
            if self.indices is not None:
                self.dialogs = [self.dialogs[idx] for idx in self.indices]
            self.dialogs = dialogs
            self.utters = []
            self.contexts = []

            for dialog in tqdm(self.dialogs, desc="Loading dialogs"):
                context = []
                for utter in dialog:
                    utter = utter.strip()
                    n_words = sum(len(w) > 2 for w in utter.split())
                    if context and min_length <= n_words <= max_length:
                        self.utters.append(utter)
                        self.contexts.append(context.copy())
                    context.append(utter)

            self.ids = list(map(str, range(len(self))))

            self.image_like_flags = [None] * len(self)
            self.image_scores = [None] * len(self)
            self.image_indices = [None] * len(self)
            self.top_n_images = [None] * len(self)

        self.path2features = path2features

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.image_dataset = None
        self.feature_paths = [None] * len(self)

        if self.path2features is not None:
            self._load_feature_vectors(
                self.path2features, model=self.model,
                tokenizer=self.tokenizer, device=device
            )

    def __getitem__(self, idx: int) -> dict:
        item = {'idx': idx, 'id': self.ids[idx],
                'utter': self.utters[idx],
                'context': self.contexts[idx],
                'image_like': self.image_like_flags[idx],
                'image_score': self.image_scores[idx]}
        if self.path2features is not None:
            item['path2features'] = self.feature_paths[idx]
            item['features'] = torch.load(self.feature_paths[idx]) if self.feature_paths[idx] is not None else None
        
        item['image_idx'] = self.image_indices[idx]
        if self.image_dataset is not None:
            item['image_dict'] = self.image_dataset[self.image_indices[idx]]

        if self.top_n_images is not None:
            if self.top_n_images[idx] is not None:
                item['top_n_images'] = self.top_n_images[idx]
        return item

    def __len__(self) -> int:
        return len(self.utters)

    def to_json(self, path: str, top_n_images=False) -> None:
        """Save dataset to json file.

        Args:
            path (str): Path to json file.
        """
        items = []
        for idx in range(len(self)):
            item = {
                'id': self.ids[idx],
                'context': self.contexts[idx], 'utter': self.utters[idx],
                'image_like': self.image_like_flags[idx],
                'image_idx': self.image_indices[idx],
                'image_score': self.image_scores[idx]
            }
            if top_n_images:
                
                item['top_n_images'] = self.top_n_images[idx]
    
            items.append(item)
        with open(path, 'w') as f:
            json.dump(items, f, indent=4)

    def from_json(self, path: str, indices: List[int] = None) -> None:
        """Load dataset from json file.

        Args:
            path (str): Path to json file.
            indices (List[int], optional):
                Indices of the dataset to load. Defaults to None.
        """
        self.dialogs, self.contexts, self.utters, self.ids = [], [], [], []
        self.image_like_flags, self.image_indices, self.image_scores = [], [], []
        self.top_n_images = []
        with open(path, 'r') as f:
            items = list(json.load(f))
        if indices is not None:
            items = [items[idx] for idx in indices]
        for i, item in enumerate(items):
            self.contexts.append(item['context'])
            self.utters.append(item['utter'])
            self.dialogs.append(item['context'] + [item['utter']])
            if 'image_like' in item:
                self.image_like_flags.append(item['image_like'])
            else:
                self.image_like_flags.append(None)
            if 'image_idx' in item:
                self.image_indices.append(item['image_idx'])
            else:
                self.image_indices.append(None)
            if 'image_score' in item:
                self.image_scores.append(item['image_score'])
            else:
                self.image_scores.append(None)
            if 'id' in item:
                self.ids.append(item['id'])
            else:
                self.ids.append(str(i))
            if 'top_n_images' in item:
                self.top_n_images.append(item['top_n_images'])
            else:
                self.top_n_images.append(None)
            

    def get_feature_vectors(self, max_workers: int = 16) -> None:
        """Load feature vector tensors from disk.

        Args:
            max_workers (int, optional): Max number of workers to spawn. Defaults to 16.
        """
        feature_vectors = [None] * len(self)
        thread_map(
            partial(self._get_feature_vector, feature_vectors=feature_vectors),
            list(range(len(self))), max_workers=max_workers,
            desc="Loading feature vector paths"
        )
        return torch.stack(feature_vectors)

    def _get_feature_vector(self, idx: int, feature_vectors: List[torch.Tensor]) -> None:
        """Helper function to load one feature vector from disk.

        Args:
            idx (int): _description_
            feature_vectors (List[torch.Tensor]): List of all feature vectors.
        """
        feature_vectors[idx] = torch.load(self.feature_paths[idx])

    def find_closest_images(
        self, image_dataset: ImageDataset, device: Union[str, torch.device] = 'cpu',
        parallel: bool = True, max_workers: int = None
    ):
        """Find closest image for each utterance in the dataset.

        Args:
            image_dataset (ImageDataset): Image dataset to search in.
            device (Union[str, torch.device], optional): PyTorch device. Defaults to 'cpu'.
            parallel (bool, optional):
                If True loads feature vectors in parallel. Defaults to True.
            max_workers (int, optional):
                Max number of workers to spawn when loading in parallel. Defaults to None.
        """
        self.image_dataset = image_dataset
        image_feature_vectors = self.image_dataset.get_feature_vectors(
            parallel=parallel, max_workers=max_workers
        ).to(device)

        for idx, path in enumerate(tqdm(self.feature_paths, desc="Finding closest images")):
            text_feature_vector = torch.load(path).to(device)
            similarities = image_feature_vectors.matmul(text_feature_vector)
            sim, image_idx = torch.max(similarities, dim=0)
            self.image_scores[idx] = sim.item()
            self.image_indices[idx] = image_idx.item()

    def find_n_closest_images(
        self, image_dataset: ImageDataset, device: Union[str, torch.device] = 'cpu',
        n = 10,
        parallel: bool = True, max_workers: int = None
    ):
        """Find closest image for each utterance in the dataset.

        Args:
            image_dataset (ImageDataset): Image dataset to search in.
            device (Union[str, torch.device], optional): PyTorch device. Defaults to 'cpu'.
            parallel (bool, optional):
                If True loads feature vectors in parallel. Defaults to True.
            max_workers (int, optional):
                Max number of workers to spawn when loading in parallel. Defaults to None.
        """
        self.image_dataset = image_dataset
        image_feature_vectors = self.image_dataset.get_feature_vectors(
            parallel=parallel, max_workers=max_workers
        ).to(device)

        for idx, path in enumerate(tqdm(self.feature_paths, desc="Finding closest images")):
            text_feature_vector = torch.load(path).to(device)
            similarities = image_feature_vectors.matmul(text_feature_vector)
            sim, image_idx = torch.topk(similarities, n, dim=0)
            self.top_n_images[idx] = image_idx.tolist()

    def _load_feature_vectors(
        self, path2dir: str, model: CLIPModel = None,
        tokenizer: CLIPTokenizer = None, device: Union[str, torch.device] = 'cpu'
    ):
        """Helper function to load feature vector paths and generate missing features.

        Args:
            path2dir (str): Path to directory containing feature tensors.
            model (CLIPModel, optional): Huggingface CLIP model. Defaults to None.
            tokenizer (CLIPTokenizer, optional): Huggingface CLIP tokenizer. Defaults to None.
            device (Union[str, torch.device], optional): PyTorch device. Defaults to 'cpu'.

        Raises:
            ValueError: If missing model or tokenizer to generate missing feature vectors.
        """
        missing_indices = []
        for idx, id in enumerate(self.ids):
            path = f"{os.path.join(path2dir, str(id))}.pt"
            if os.path.exists(path):
                self.feature_paths[idx] = path
            else:
                missing_indices.append(idx)

        if missing_indices:
            if model is None or tokenizer is None:
                raise ValueError("Missing feature vectors, but no model or tokenizer for generation")
            self._generate_feature_vectors(
                missing_indices, model, tokenizer, path2dir, device
            )

    def _generate_feature_vectors(
        self, indices: List[int], model: CLIPModel,
        tokenizer: CLIPTokenizer, path2dir: str, device: Union[str, torch.device] = 'cpu'
    ):
        """Helper function to generate utterance feature vectors using a CLIP model.

        Args:
            indices (List[int]): Indices of the dataset to generate feature vectors for.
            model (CLIPModel): Huggingface CLIP model.
            tokenizer (CLIPTokenizer): Huggingface CLIP tokenizer.
            path2dir (str): Path to directory containing feature tensors.
            device (Union[str, torch.device], optional): PyTorch device. Defaults to 'cpu'.
        """
        dataloader = DataLoader(
            Subset(self, indices), batch_size=256,
            shuffle=False, collate_fn=collate_fn
        )
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating feature vectors"):
                utters = batch['utters']
                inputs = tokenizer(text=utters, padding=True, truncation=True, return_tensors="pt").to(device)
                features = model.get_text_features(**inputs)
                features /= features.norm(dim=-1, keepdim=True)
                for vector, idx, id in zip(features, batch['indices'], batch['ids']):
                    path = f"{os.path.join(path2dir, str(id))}.pt"
                    torch.save(vector.to('cpu'), path)
                    self.feature_paths[idx] = path


def collate_fn(data: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Function to collate lists of data in a batch.

    Args:
        data (Dict[str, List[Any]]): Lists of data.

    Returns:
        Dict[str, Any]: Batach of data.
    """
    batch = {batch_key: [item[item_key] for item in data] for batch_key, item_key in [
                 ('indices', 'idx'), ('ids', 'id'), ('utters', 'utter'), ('contexts', 'context')
            ]}
    for batch_key, item_key in [
        ('feature_paths', 'path2features'), ('features', 'features')
    ]:
        if item_key in data[0]:
            batch[batch_key] = [item[item_key] for item in data]
    return batch
