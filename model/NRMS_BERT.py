import torch
from torch import nn
import torch.nn.functional as F

from .model_utils import AttentionPooling, MultiHeadSelfAttention
from transformers import BertModel, BertTokenizer, BertConfig


model_name_or_path = "/home/sunwenqi/pretrained_model/bert-base-uncased"

torch.cuda.empty_cache()

class NewsEncoder(nn.Module):
    def __init__(self, args, embedding_matrix):
        super(NewsEncoder, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.drop_rate = args.drop_rate
        self.dim_per_head = args.news_dim // args.num_attention_heads
        assert args.news_dim == args.num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = MultiHeadSelfAttention(
            args.word_embedding_dim,
            args.num_attention_heads,
            self.dim_per_head,
            self.dim_per_head
        )
        self.attn = AttentionPooling(args.news_dim, args.news_query_vector_dim)

    def forward(self, x, mask=None):
        '''
            x: batch_size, word_num
            mask: batch_size, word_num
        '''
        model = BertModel.from_pretrained(model_name_or_path)
        model = model.cuda(0)
        print("--x--")
        print(x)
        print(x.shape)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        sequence = model(x.long())[0]
        word_vecs = F.dropout(sequence,
                              p=self.drop_rate,
                              training=self.training)
        print(word_vecs)
        print(word_vecs.shape)
        multihead_text_vecs = self.multi_head_self_attn(word_vecs, word_vecs, word_vecs, mask)
        multihead_text_vecs = F.dropout(multihead_text_vecs,
                                        p=self.drop_rate,
                                        training=self.training)
        news_vec = self.attn(multihead_text_vecs, mask)
        return news_vec


class UserEncoder(nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.dim_per_head = args.news_dim // args.num_attention_heads
        assert args.news_dim == args.num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = MultiHeadSelfAttention(args.news_dim, args.num_attention_heads, self.dim_per_head, self.dim_per_head)
        self.attn = AttentionPooling(args.news_dim, args.user_query_vector_dim)
        self.pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)

    def forward(self, news_vecs, log_mask=None):
        '''
            news_vecs: batch_size, history_num, news_dim
            log_mask: batch_size, history_num
        '''
        bz = news_vecs.shape[0]
        if self.args.user_log_mask:
            news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs, log_mask)
            user_vec = self.attn(news_vecs, log_mask)
        else:
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.args.user_log_length, -1)
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1))
            news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs)
            user_vec = self.attn(news_vecs)
        return user_vec


class Model(torch.nn.Module):
    def __init__(self, args, embedding_matrix, num_category, num_subcategory, **kwargs):
        super(Model, self).__init__()
        self.args = args
        pretrained_word_embedding = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding,
                                                      freeze=args.freeze_embedding,
                                                      padding_idx=0)

        self.news_encoder = NewsEncoder(args, word_embedding)
        self.user_encoder = UserEncoder(args)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, history, history_mask, candidate, label):
        '''
            history: batch_size, history_length, num_word_title
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title
            label: batch_size, 1+K
        '''
        print("--candidate--")
        print(candidate)
        print(candidate.shape)
        candidate_news = candidate.reshape(-1, self.args.num_words_title)
        #print(candidate_news)
        #print(candidate_news.shape)
        candidate_news_vecs = self.news_encoder(candidate_news.long())
        #print(candidate_news_vecs)
        #print(candidate_news_vecs.shape)
        # print(candidate_news_vecs.shape)
        #candidate_news_vecs = candidate_news_vecs[:175,]
        candidate_news_vecs = candidate_news_vecs.reshape(-1, 1 + self.args.npratio, self.args.news_dim)
        #print(candidate_news_vecs)
        #print(candidate_news_vecs.shape)
        #print("--candidate_news_vecs--")
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        print(history_news)
        print(history_news.shape)
        history_news = history.reshape(-1, self.args.num_words_title)
        print(history_news)
        history_news_vecs = self.news_encoder(history_news)
        # if history_news_vecs.size % (self.args.user_log_length*self.args.news_dim) !=0:
        #    history_news_vecs = history_news_vecs[ : history_news_vecs.size / (self.args.user_log_length*self.args.news_dim)* (self.args.user_log_length*self.args.news_dim)]
        #print(history_news_vecs)
        #print(history_news_vecs.shape)
        #history_news_vecs = history_news_vecs[:1750,]
        history_news_vecs = history_news_vecs.reshape(-1, self.args.user_log_length, self.args.news_dim)

        user_vec = self.user_encoder(history_news_vecs, history_mask)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
        return loss, score