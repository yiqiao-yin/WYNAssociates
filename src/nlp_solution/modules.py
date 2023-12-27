import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel


class nlp:
    def get_models_():
        """get the model for bert tokenizer and the model"""
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = TFBertModel.from_pretrained("bert-base-uncased")
        return tokenizer, model

    def bert_encoder_(text: str):
        """bert encoder: a unit function to embed one string"""
        encoded_input = tokenizer(text, return_tensors="tf")
        output = model(encoded_input)
        return np.asarray(output[0]).sum()

    def sort_names_(names: np.ndarray):
        """embed an array of names through the bert space"""
        results_ = []
        for i in tqdm(range(names.shape[0])):
            results_.append(
                [
                    names[i, 0],
                    names[i, 1],
                    bert_encoder_(names[i, 0]),
                    bert_encoder_(names[i, 1]),
                    bert_encoder_(names[i, 2]),
                ]
            )

        results_ = np.asarray(results_)
        results_ = pd.DataFrame(results_[np.argsort(results_[:, 3]), :]).iloc[:, 0:2]

        return results_
