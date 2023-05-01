from functools import partial
import os
import json
from typing import List, Union, Tuple, Dict, Any

import requests
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm.contrib.concurrent import thread_map
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPFeatureExtractor
import pandas as pd


class ImageDataset(Dataset):
    """A class containing an image dataset and its associated metadata.
    """
    def __init__(
        self, photos_df: Union[str, pd.DataFrame], path2images: str = None, width: int = 640,
        drop_missing: bool = True, path2features: str = None, model: CLIPModel = None,
        feature_extractor: CLIPFeatureExtractor = None, device: Union[str, torch.device] = 'cpu',
        check_image_integrity: bool = True, parallel: bool = False, max_workers: int = None,
        indices: List[int] = None
    ) -> None:
        """
        Args:
            photos_df (Union[str, pd.DataFrame]):
                Either a path to a json file or a Pandas DataFrame containing the dataset.
            path2images (str, optional):
                Path to directory containing saved images. Defaults to None.
            width (int, optional):
                Width of a downloaded image. Defaults to 640.
            drop_missing (bool, optional):
                If True missing images are dropped. Defaults to True.
            path2features (str, optional):
                Path to directory containing saved features. Defaults to None.
            model (CLIPModel, optional):
                Huggingface CLIP model. Defaults to None.
            feature_extractor (CLIPFeatureExtractor, optional):
                Huggingface CLIP feature extractor. Defaults to None.
            device (Union[str, torch.device], optional):
                PyTorch device. Defaults to 'cpu'.
            check_image_integrity (bool, optional):
                If True each images is opened when loading. Defaults to True.
            parallel (bool, optional):
                If True loads images in parallel. Defaults to False.
            max_workers (int, optional):
                Max number of workers to spawn. Defaults to None. Defaults to None.
            indices (List[int], optional):
                Indices of DataFrame or json to include in the dataset. Defaults to None.
        """
        self.photos_df = photos_df
        self.indices = indices

        if isinstance(self.photos_df, str) and self.photos_df.endswith('json'):
            self.from_json(self.photos_df, indices=indices)
        else:
            if self.indices is not None:
                self.photos_df = self.photos_df.iloc[self.indices]
            self.ids = self.photos_df.photo_id.to_list()
            self.urls = [
                f"{url}?w={width}" for url in self.photos_df.photo_image_url]
            self.descriptions = self.photos_df.photo_description.to_list()
            self.ai_descriptions = self.photos_df.ai_description.to_list()

        self.image_paths = [None] * len(self)
        self.feature_paths = [None] * len(self)

        self.path2images = path2images
        self.path2features = path2features
        self.feature_vectors = None

        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.check_image_integrity = check_image_integrity

        if self.path2images is not None:
            self._load_images(self.path2images, drop_missing=drop_missing,
                              parallel=parallel, max_workers=max_workers,
                              check_image_integrity=check_image_integrity)
        if self.path2features is not None:
            self._load_feature_vectors(
                self.path2features, model=self.model,
                feature_extractor=self.feature_extractor, device=device
            )

    def __getitem__(self, idx: int) -> None:
        item = {'idx': idx, 'id': self.ids[idx], 'url': self.urls[idx],
                'description': self.descriptions[idx],
                'ai_description': self.ai_descriptions[idx]}
        if self.path2images is not None:
            item['path2image'] = self.image_paths[idx]
            item['image'] = Image.open(
                self.image_paths[idx]) if self.image_paths[idx] is not None else None
        if self.path2features is not None:
            item['path2features'] = self.feature_paths[idx]
            item['features'] = torch.load(self.feature_paths[idx]) if self.feature_paths[idx] is not None else None
        return item

    def __len__(self):
        return len(self.ids)

    def to_json(self, path: str) -> None:
        """Save dataset to json file.

        Args:
            path (str): Path to json file.
        """
        items = []
        for idx in tqdm(range(len(self)), desc="Saving to json"):
            item = {
                'id': self.ids[idx], 'url': self.urls[idx],
                'description': self.descriptions[idx],
                'ai_description': self.ai_descriptions[idx],
            }
            items.append(item)
        with open(path, 'w') as f:
            json.dump(items, f, indent=4)

    def from_json(self, path: str, indices: List[int] = None):
        """Load dataset from json file.

        Args:
            path (str): Path to json file.
            indices (List[int], optional):
                Indices of the dataset to load. Defaults to None.
        """
        self.ids, self.urls = [], []
        self.descriptions, self.ai_descriptions = [], []
        with open(path, 'r') as f:
            items = list(json.load(f))
        if indices is not None:
            items = [items[idx] for idx in indices]
        for item in tqdm(items, desc='Reading from json'):
            self.ids.append(item['id'])
            self.urls.append(item['url'])
            self.descriptions.append(item['description'])
            self.ai_descriptions.append(item['ai_description'])

    def get_feature_vectors(self, parallel: bool = True, max_workers: int = None) -> None:
        """Load feature vectors from disk

        Args:
            parallel (bool, optional):
                If True loads images in parallel. Defaults to True.
            max_workers (int, optional):
                Max number of workers to spawn. Defaults to None.. Defaults to None.
        """
        if self.feature_vectors is None:
            feature_vectors = [None] * len(self)
            if not parallel:
                for idx in tqdm(range(len(self)), desc="Getting feature vectors"):
                    self._get_feature_vector(idx, feature_vectors=feature_vectors)
            else:
                thread_map(
                    partial(self._get_feature_vector, feature_vectors=feature_vectors),
                    list(range(len(self))), max_workers=max_workers,
                    desc="Getting feature vectors"
                )
            self.feature_vectors = torch.stack(feature_vectors)
        return self.feature_vectors

    def _get_feature_vector(self, idx: int, feature_vectors: List[torch.Tensor]) -> None:
        """Helper function to load one feature vector from disk.

        Args:
            idx (int): Index of the image.
            feature_vectors (List[torch.Tensor]): List of all feature vectors.
        """
        feature_vectors[idx] = torch.load(self.feature_paths[idx])

    def _load_images(
        self, path2dir: str, drop_missing: bool = False, max_workers: int = None,
        check_image_integrity: bool = True, parallel: bool = False
    ) -> None:
        """Helper function to load image paths from disk and download missing from urls.

        Args:
            path2dir (str):
                Path to directory containing the images.
            drop_missing (bool, optional):
                If True missing images are dropped. Defaults to False.
            max_workers (int, optional):
                Max number of workers to spawn. Defaults to None.
            check_image_integrity (bool, optional):
                If True each images is opened when loading. Defaults to True.
            parallel (bool, optional):
                If True uses multiprocessing to load image paths. Defaults to False.
        """
        if not parallel:
            for args in tqdm(zip(range(len(self)), self.ids, self.urls),
                             total=len(self), desc="Loading image paths"):
                self._load_image(args, path2dir=path2dir,
                                 check_image_integrity=check_image_integrity)
        else:
            thread_map(
                partial(self._load_image, path2dir=path2dir,
                        check_image_integrity=check_image_integrity),
                zip(range(len(self)), self.ids, self.urls),
                total=len(self), max_workers=max_workers,
                desc="Loading image paths"
            )
        print(f"Failed to load {self.image_paths.count(None)}/{len(self)} images")

        if drop_missing:
            missing_indices = [idx for idx, path in enumerate(self.image_paths) if path is None]
            for idx in sorted(missing_indices, reverse=True):
                for lst in [self.ids, self.urls, self.descriptions, self.ai_descriptions, self.image_paths]:
                    lst.pop(idx)

    def _load_image(
        self, args: Tuple[int, str, str], path2dir: str, check_image_integrity: bool = True
    ) -> None:
        """Helper function to load one image.

        Args:
            args (Tuple):
                Tuple containing index, id and url.
            path2dir (str):
                Path to directory containing the images.
            check_image_integrity (bool, optional):
                If True each images is opened when loading. Defaults to True.
        """
        idx, id, url = args
        path = f"{os.path.join(path2dir, id)}.jpg"
        if not os.path.exists(path):
            try:
                self._download_image(url).save(path)
            except Exception:
                # print(f"Can't load image {id} from {url}")
                return
        if check_image_integrity:
            try:
                Image.open(path)
            except Exception:
                # print(f"Corrupted image file {path}")
                return
        self.image_paths[idx] = path

    def _download_image(self, url: str) -> None:
        """Helper function to load one image from its url

        Args:
            url (str): Url to load image from.
        """
        return Image.open(requests.get(url, stream=True).raw).convert('RGB')

    def _load_feature_vectors(
        self, path2dir: str, model: CLIPModel = None,
        feature_extractor: CLIPFeatureExtractor = None, device: Union[str, torch.device] = 'cpu'
    ) -> None:
        """Helper function to load feature vector paths and generate missing.

        Args:
            path2dir (str): Path to directory containing feature vectors.
            model (CLIPModel, optional):
                Huggingface CLIP model. Defaults to None.
            feature_extractor (CLIPFeatureExtractor, optional):
                Huggingface CLIP feature extractor. Defaults to None.
            device (Union[str, torch.device], optional):
                PyTorch device. Defaults to 'cpu'.

        Raises:
            ValueError: If missing model or feature extractor to load missing feature vectors.
        """
        missing_indices = []
        for idx, id in enumerate(tqdm(self.ids, desc="Loading feature vector paths")):
            path = f"{os.path.join(path2dir, id)}.pt"
            if os.path.exists(path):
                self.feature_paths[idx] = path
            else:
                missing_indices.append(idx)

        if missing_indices:
            if model is None or feature_extractor is None:
                raise ValueError("Missing feature vectors, but no model or feature extractor for generation")
            self._generate_feature_vectors(
                missing_indices, model, feature_extractor, path2dir, device
            )

    def _generate_feature_vectors(
        self, indices: List[int], model: CLIPModel,
        feature_extractor: CLIPFeatureExtractor, path2dir: str, device: Union[str, torch.device] = 'cpu'
    ) -> None:
        """Helper function to generate feature vectors.

        Args:
            indices (List[int]):
                Indices of dataset to generate feature vectors for.
            model (CLIPModel):
                Huggingface CLIP model.
            feature_extractor (CLIPFeatureExtractor):
                Huggingface CLIP feature extractor.
            path2dir (str):
                Path to directory containing feature tensors.
            device (Union[str, torch.device], optional):
                PyTorch device. Defaults to 'cpu'.
        """
        dataloader = DataLoader(Subset(self, indices), batch_size=128, shuffle=False, collate_fn=collate_fn)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating feature vectors"):
                images = batch['images']
                inputs = feature_extractor(images=images, return_tensors="pt").to(device)
                features = model.get_image_features(**inputs)
                features /= features.norm(dim=-1, keepdim=True)
                for vector, idx, id in zip(features, batch['indices'], batch['ids']):
                    path = f"{os.path.join(path2dir, id)}.pt"
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
                 ('indices', 'idx'), ('ids', 'id'), ('urls', 'url'),
                 ('descriptions', 'description'), ('ai_descriptions', 'ai_description')
            ]}
    for batch_key, item_key in [
        ('image_paths', 'path2image'),
        ('images', 'image'), ('feature_paths', 'path2features'),
        ('features', 'features')
    ]:
        if item_key in data[0]:
            batch[batch_key] = [item[item_key] for item in data]
    return batch
