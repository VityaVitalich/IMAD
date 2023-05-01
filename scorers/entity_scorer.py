from .schemas import ModelInitMixin, Paths
import pandas as pd
from data_utils.dialog_utils import DialogDataset
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


class EntityScorer(ModelInitMixin):
    
    def __init__(self,
                device: str,
                path: Paths):
        
        self.device = device
        self.path = path

        if not self.check_if_inited():
            self.init_models()

    def make_entity_scoring(self):

        scored = pd.read_json(self.path.output_path)
        entity_list = self.get_entity_list(scored)
        scored['entity'] = entity_list

        self.construct_unique_entity_df(scored)
        self.score_unique_entity()


        scored_test = pd.read_json(self.path.temporary_path)
        scored_test['utter'] = scored_test['utter'].str.replace('a photo of a ', '')

        dict_scores = {}
        for i in range(len(scored_test)):
            dict_scores[scored_test['utter'][i]] = scored_test['image_score'][i]

        scores = []
        for ls in scored['entity']:
            score_ls = []
            for word in ls:
                score_ls.append(dict_scores[word])
            if len(score_ls) == 0:
                score_ls.append(0)
            scores.append(score_ls)

        scored['scores_entity'] = scores
        scored['max_entity_score'] = scored['scores_entity'].apply(max)
        scored.to_json(self.path.output_path, indent= 4, orient = 'records')

    def score_unique_entity(self):


        dialog_dataset = DialogDataset(
                            self.path.temporary_path,
                            path2features=self.path.entity_vectors_path,
                            model=self.clip_model, tokenizer=self.clip_tokenizer, device=self.device
                        )
        dialog_dataset.find_closest_images(self.image_dataset, device=self.device)
        dialog_dataset.to_json(self.path.temporary_path) 

    def construct_unique_entity_df(self, scored):

        unique_entity = []
        for elem in scored['entity'].values:
            for word in elem:
                if word not in unique_entity:
                    unique_entity.append(word)

        unique_entity_df = pd.DataFrame(unique_entity)

        unique_entity_df['context'] = ''
        unique_entity_df['context'] = unique_entity_df['context'].apply(list)
       
        unique_entity_df.columns = ['utter', 'context']
        unique_entity_df['entity'] = unique_entity_df['utter'].copy()
        unique_entity_df['utter'] = 'a photo of a ' + unique_entity_df['utter'].astype(str)
        unique_entity_df.to_json(self.path.temporary_path, orient = 'records', indent = 4) 


    def get_entity_list(self, scored):

        lemmatized_utters = []
        tagged_utters = []

        tw = TweetTokenizer()
        lemmatizer = WordNetLemmatizer()

        for utter in scored['utter']:
            utter = utter.replace('.', ' ').replace('’', '')
            tokenized_utter = tw.tokenize(utter)
            lemmatized_phrase = []
            for word in tokenized_utter:
                lemmatized_phrase.append(lemmatizer.lemmatize(word))
                
            tagged_utters.append(nltk.pos_tag(lemmatized_phrase))
            lemmatized_utters.append(lemmatized_phrase)

        entity_list = []
        for sent in tagged_utters:
            entity_sent = []
            for pair in sent:
                if pair[1] == 'NN': #or pair[1] == 'NNP':
                    entity_sent.append(pair[0])
            entity_list.append(entity_sent)

        return entity_list