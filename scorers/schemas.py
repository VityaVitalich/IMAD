from dataclasses import dataclass
from transformers import CLIPModel, CLIPTokenizer
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForImageTextRetrieval
from sentence_transformers import SentenceTransformer
from data_utils.image_utils import ImageDataset
import torch

@dataclass
class Paths:
    dialog_features_path: str = './feature_vectors/test_vectors/'
    image_vectors_path: str = './images/vectors.pt'
    output_path: str = './outputs/test_output.json'
    temporary_path: str = './outputs/temporary_path.json'
    entity_vectors_path: str = './feature_vectors/entity_vectors/'
    images_dataset_path: str =  './images/dataset.json'
    path2images: str = './images/full_images'
    path2images_features: str = './images/vectors'
    path2trained_model: str = './models/random_forest.joblib'



class ModelInitMixin:

    def init_models_clip(self) -> None:

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    def init_image_dataset(self) -> None:
        self.image_dataset = ImageDataset(
                                        self.path.images_dataset_path,
                                        path2images=self.path.path2images,
                                        path2features=self.path.path2images_features,
                                        check_image_integrity=False, parallel=False, max_workers=None
                                    )
        self.image_dataset.feature_vectors = torch.load(self.path.image_vectors_path)

    def delete(self, model: str = 'clip_model'):
        if self.check_if_inited(model):
            if model == 'clip':
                del self.clip_model
                del self.clip_tokenizer
            elif model == 'caption_model':
                del self.caption_model
                del self.caption_tokenizer
                del self.caption_feature_extractor
            

            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def check_if_inited(self, model: str = 'clip_model') -> bool:

        return hasattr(self, model)
        
    def init_captioners(self) -> None:
        self.caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(self.device)
        self.caption_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    def init_sentence_sim_model(self) -> None:
        self.ss_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(self.device)

    def init_vqa_model(self) -> None:
        self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)