from .schemas import ModelInitMixin, Paths
from data_utils.dialog_utils import DialogDataset
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np


class VQAScorer(ModelInitMixin):

    def __init__(self, path, device, n=10):

        self.path = path
        self.device = device
        self.n = n

    def find_top_n_images(self):

        if not self.check_if_inited('clip_model'):
            self.init_models_clip()
        if not self.check_if_inited('image_dataset'):
            self.init_image_dataset()

        dialog_dataset = DialogDataset(
            self.path.output_path,
            path2features=self.path.dialog_features_path,
            model=self.clip_model, tokenizer=self.clip_tokenizer, device=self.device
        )

        dialog_dataset.find_n_closest_images(self.image_dataset, device=self.device, n=self.n)
        dialog_dataset.to_json(self.path.temporary_path, top_n_images=True)

        df = pd.read_json(self.path.output_path)
        top_n_images = pd.read_json(self.path.temporary_path)

        df['top_n_images'] = top_n_images['top_n_images']
        df.to_json(self.path.output_path, indent= 4, orient = 'records')



    @staticmethod
    def _shift_right(model, input_ids):
        pad_token_id = model.decoder_pad_token_id

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = model.decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


    def get_confidence(self, question, true_utter, image):


        inputs = self.vqa_processor(image, question, return_tensors="pt").to(self.device)
        
        labels = self.vqa_processor(text = true_utter, return_tensors = "pt").to(self.device)
        labels = labels['input_ids']

        vision_outputs = self.vqa_model.vision_model(
                pixel_values=inputs['pixel_values'],
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False,
            )


        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        question_embeds = self.vqa_model.text_encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )
        question_embeds = question_embeds[0]

        self.vqa_model.decoder_pad_token_id = self.vqa_model.config.text_config.pad_token_id
        self.vqa_model.decoder_start_token_id = self.vqa_model.config.text_config.bos_token_id

        decoder_input_ids = self._shift_right(self.vqa_model, labels)

        labels = labels.masked_fill(labels == self.vqa_model.decoder_pad_token_id, -100)

        answer_output = self.vqa_model.text_decoder(
            input_ids=decoder_input_ids.cpu(),
            encoder_hidden_states=question_embeds.to(self.device),
            encoder_attention_mask=inputs['attention_mask'].to(self.device),
            labels=labels.to(self.device),
            return_dict=torch.tensor(True).to(self.device),
            reduction="mean",
        )

        logits = answer_output['logits']
        log_proba = F.log_softmax(logits, dim=-1)

        true_proba = log_proba[torch.arange(log_proba.shape[0]).unsqueeze(-1),
                                    torch.arange(log_proba.shape[1]).unsqueeze(0),
                                    labels]

        confidences = true_proba.mean(dim=1).cpu().detach().numpy().tolist()

        return confidences

    def get_vqa_images(self):

        if not self.check_if_inited('vqa_model'):
            self.init_vqa_model()
            self.delete('clip_model')
        if not self.check_if_inited('image_dataset'):
            self.init_image_dataset()

        self.df = pd.read_json(self.path.output_path)

        conf_ls = []
        question = 'What phrase can describe this picture?'


        for idx in tqdm(range(len(self.df))):
            true_utter = self.df['utter'][idx]
            images_idx = self.df['top_n_images'][idx]
            images = [self.image_dataset[elem]['image'] for elem in images_idx]

            questions = [question] * self.n
            true_utter = [true_utter] * self.n 

            try:
                conf_obj = self.get_confidence(questions, true_utter, images)
            except RuntimeError:
                print('Runtime Error at ' + str(idx))
                conf_obj = [1]

            conf_ls.append(conf_obj)

            del conf_obj
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        self.df['confidence_vqa'] = conf_ls

        self.df['vqa_idx'] = self.df['confidence_vqa'].apply(np.argmax)
        idxes = np.vstack(self.df['top_n_images'].values)
        self.df['vqa_image'] = idxes[np.arange(len(idxes)), self.df['vqa_idx'].values]

        self.df.to_json(self.path.output_path, indent= 4, orient = 'records')


