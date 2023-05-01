from .schemas import ModelInitMixin, Paths
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader
from data_utils.dialog_utils import DialogDataset
import pandas as pd
from tqdm import tqdm
import numpy as np


class Captioner(ModelInitMixin):

    def __init__(self, device: str, path: Paths):
        self.path = path
        self.device = device
        self.delete_previous_model = True
        self.config = {'BS': 16,
                       "gen_kwargs": {
                            "max_length": 25,
                            "num_beams": 10
                                    }
                        }

    def make_captions(self) -> None:


        if self.delete_previous_model:
            self.delete(model='clip_model')

        self.init_captioners()

        df = pd.read_json(self.path.output_path)
        dialog_dataset = DialogDataset(self.path.output_path)
        loader = DataLoader(dialog_dataset, batch_size = self.config['BS'], collate_fn = self.collate_fn)


        df['captioned_text'] = ''
        start_idx = 0
        end_idx = self.config['BS']
        iters = 0
        for batch in tqdm(loader):
            
            pixel_values = self.caption_feature_extractor(images=batch, return_tensors="pt").pixel_values.to(self.device)

            output_ids = self.caption_model.generate(pixel_values, **self.config['gen_kwargs'])

            preds = self.caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            df['captioned_text'][start_idx:end_idx] = preds

            start_idx = end_idx
            end_idx += self.config['BS'] 
            iters += 1

            if iters % 2 == 0:
                df.to_json(self.path.output_name, indent= 4, orient = 'records')


        df.to_json(self.path.output_path, indent= 4, orient = 'records')

    def collate_fn(self, batch):


        img_ls = []
        for obj in batch:
            idx = obj['image_idx']
            img = self.image_dataset[idx]['image']
            img_ls.append(img)

        return img_ls
    

class SentenceSimilarityScorer(ModelInitMixin):

    def __init__(self,
                 device: str,
                 path: Paths, 
                 bs=2048):
        
        self.bs = bs

    def make_sentence_similarities(self) -> None:
        
        df = pd.read_json(self.path.output_path)

        if not self.check_if_inited('ss_model'):
            self.delete('caption_model')
            self.init_sentence_sim_model()

        cos_sim = self.get_cosine_sim(df)
        df['sent_sim'] = cos_sim
        df.to_json(self.path.output_path, indent= 4, orient = 'records')


    def get_cosine_sim(self, df):

        cos_sim = self.get_embeddings(df['utter'], df['captioned_text'])
        return cos_sim

    def get_embeddings(self, df_col_utter, df_col_captioned):
        values_utter = df_col_utter.values
        values_captioned = df_col_captioned.values

        split_size = max(1, int(values_utter.shape[0] / self.bs))

        chunks_utter = np.array_split(values_utter, split_size)
        chunks_captioned = np.array_split(values_captioned, split_size)

        res = []
        for i in tqdm(range(len(chunks_utter))):
            utter_emb = (self.ss_model.encode(chunks_utter[i]))
            cap_emb = (self.ss_model.encode(chunks_captioned[i]))

            d = pairwise_distances(utter_emb, cap_emb, metric='cosine')
            cos_sim = 1 - np.diagonal(d)

            res.append(cos_sim)

        return np.hstack(res)