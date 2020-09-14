from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BERT_CONFIG_PATH = "/Users/zhangwanyu/bert-base-chinese/bert_config.json"
BERT_CHECKPOINT_PATH = "/Users/zhangwanyu/bert-base-chinese/bert_model.ckpt"
BERT_VOCAB_PATH = "/Users/zhangwanyu/bert-base-chinese/vocab.txt"


class MaskedLM(object):
    def __init__(self,topK):
        self.topK = topK
        self.tokenizer = Tokenizer(BERT_VOCAB_PATH,do_lower_case='True')
        self.model = build_transformer_model(BERT_CONFIG_PATH,BERT_CHECKPOINT_PATH,with_mlm=True)

    def tokenizer_text(self,text):
        # ['[CLS]', '我', '喜', '欢', '吃', '程', '度', '的', '火', '锅', '[SEP]']
        self.toeken = self.tokenizer.tokenize(text)
        # [101, 2769, 1599, 3614, 1391, 4923, 2428, 4638, 4125, 7222, 102] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.token_ids, self.segment_ids = self.tokenizer.encode(text)

    def find_top_candidates(self,error_index):
        for i in error_index:
            # 将错误词的id换成[MASK]的id
            self.token_ids[i] = self.tokenizer._token_dict['[MASK]']
        # 第 5，6 个位置被替换为mask的ID-103，[101, 2769, 1599, 3614, 1391, 103, 103, 4638, 4125, 7222, 102]
        # 预测每一个token的概率分布 probs.shape = [len(toekn_ids),vocab_size]
        probs = self.model.predict([np.array([self.token_ids]),np.array([self.segment_ids])])[0]

        for i in range(len(error_index)):
            # 拿到error_id
            error_id = error_index[i]
            # 取出概率分布里面，概率最大的topK个的位置id,argsort是升序，取负之后倒序
            top_k_probs = np.argsort(-probs[error_id])[:self.topK]
            candidates,find_prob = self.tokenizer.decode(top_k_probs),probs[error_id][top_k_probs]
            print(dict(zip(candidates,find_prob)))

if __name__ == '__main__':
    maskLM = MaskedLM(5)
    text = '我喜欢吃程度的火锅'
    maskLM.tokenizer_text(text)
    maskLM.find_top_candidates([5,6])