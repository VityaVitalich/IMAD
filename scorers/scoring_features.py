from typing import List
from data_utils.dialog_utils import DialogDataset
import pandas as pd
import numpy as np
from sacrebleu.metrics import BLEU
import joblib
from .schemas import ModelInitMixin, Paths
from .caption_features import Captioner, SentenceSimilarityScorer
from .entity_scorer import EntityScorer


class FeatureCreator(EntityScorer, Captioner, SentenceSimilarityScorer):

    def __init__(self,
                examples: List[List[str]],
                device: str,
                path: Paths) -> None:

        Captioner.__init__(self, device, path)
        SentenceSimilarityScorer.__init__(self, device, path)
        self.device = device
        self.path = path
        self.examples = examples




    def get_ImageScore(self) -> None:
        
        if not self.check_if_inited('clip_model'):
            self.init_models_clip()
        if not self.check_if_inited('image_dataset'):
            self.init_image_dataset()

        dialog_dataset = DialogDataset(
            self.examples,
            path2features=self.path.dialog_features_path,
            model=self.clip_model, tokenizer=self.clip_tokenizer, device=self.device
        )
        dialog_dataset.find_closest_images(self.image_dataset, device=self.device)
        dialog_dataset.to_json(self.path.output_path) 

    def get_entity_scores(self) -> None:

        if not self.check_if_inited('clip_model'):
            self.init_models_clip()
        if not self.check_if_inited('image_dataset'):
            self.init_image_dataset()

        self.make_entity_scoring()

    def get_captions(self) -> None:
        
        if not self.check_if_inited('image_dataset'):
            self.init_image_dataset()
            
        self.make_captions()

    def get_sentence_similarity(self) -> None:

        self.make_sentence_similarities()

    def get_bleu_score(self) -> None:

        bleu = BLEU(effective_order=True)
        df = pd.read_json(self.path.output_path)

        bleu_ls = []

        for i in range(len(df)):
            hyp = df['captioned_text'][i]
            ref = df['utter'][i]

            bleu_sent = bleu.sentence_score(hyp, [ref])
            bleu_ls.append(bleu_sent.score)

        df['bleu_1ngram'] = bleu_ls

        df.to_json(self.path.output_path, indent=4, orient='records')

    def get_thresholding(self) -> None:

        df = pd.read_json(self.path.output_path)

        MES_tr = 0.3103302687291667
        IS_tr = 0.33265801843083337
        SS_tr = 0.12116438820166667

        threshold = (df['max_entity_score'] > MES_tr) & (df['image_score'] > IS_tr) & (df['sent_sim'] > SS_tr)
        df['threshold'] = threshold
        df.to_json(self.path.output_path, indent=4, orient='records')

class ImageReplacityScorer():

    def __init__(self, path: Paths):
        self.path = path
        self.model = joblib.load(self.path.path2trained_model)

        self.thresholds = {8: 0.027817989779176946,
                     7: 0.050168573450116334,
                     6: 0.07258147026428512,
                     5: 0.12009801234825425,
                     4: 0.18145667982860553,
                     3: 0.26931306865778737,
                     2: 0.404215777212629,
                     1: 0.5508405776200066}

    def predict_proba(self, X_test):
        y_pred_proba = self.model.predict_proba(X_test)
        return y_pred_proba

    def make_predictions(self, threshold=8):

        if threshold in self.thresholds.keys():
            threshold = self.thresholds[threshold]
        else:
            assert type(threshold) == float
        
        df = pd.read_json(self.path.output_path)
        X_test = df[['image_score', 'bleu_1ngram', 'sent_sim', 'max_entity_score', 'threshold']]

        y_pred_proba = self.predict_proba(X_test)
        y_pred = y_pred_proba[:,1] >= threshold
        df['image_like'] = y_pred
        df.to_json(self.path.output_path, indent=4, orient='records')

        
        


