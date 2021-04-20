import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, AlbertModel, AutoConfig

class RedditAlbertModel(nn.Module):
    def __init__(self, hyp_params):
        super(RedditAlbertModel, self).__init__()
        self.bert = AlbertModel.from_pretrained(hyp_params.bert_model)
        self.drop = nn.Dropout(p=0.1)
        #self.out = nn.Linear(hyp_params.bert_hidden_size, hyp_params.output_dim)
        self.out = nn.Linear(hyp_params.bert_hidden_size, 512)
        self.out2 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(outputs.pooler_output)
        output = self.out(output)
        output = self.out2(output)
        return output

class RedditBertModel(nn.Module):
    def __init__(self, hyp_params):
        super(RedditBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(hyp_params.bert_model)
        self.drop = nn.Dropout(p=0.1)
        #self.out = nn.Linear(hyp_params.bert_hidden_size, hyp_params.output_dim)
        self.out = nn.Linear(hyp_params.bert_hidden_size, 512)
        self.out2 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(outputs.pooler_output)
        output = self.out(output)
        output = self.out2(output)
        return output

class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        # weight for textual feature: w_t
        weights_hidden1 = torch.Tensor(size_out, size_in1)
        self.weights_hidden1 = nn.Parameter(weights_hidden1)

        # weight for visual feature: w_i
        weights_hidden2 = torch.Tensor(size_out, size_in2)
        self.weights_hidden2 = nn.Parameter(weights_hidden2)

        # Weight for sigmoid: w_h
        weight_sigmoid = torch.Tensor(size_out*2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_t, x_i):
        h_t = self.tanh_f(torch.mm(x_t, self.weights_hidden1.t()))
        h_i = self.tanh_f(torch.mm(x_i, self.weights_hidden2.t()))
        y = torch.cat((h_t, h_i), dim=1)
        z = self.sigmoid_f(torch.matmul(y, self.weight_sigmoid.t()))
        z_t = z.view(z.size()[0],1)*h_t 
        z_i = (1-z).view(z.size()[0],1)*h_i
        f = z_t + z_i
        return f


class AverageBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(AverageBERTModel, self).__init__()
        self.linear1 = nn.Linear(hyp_params.bert_hidden_size+hyp_params.image_feature_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear2 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, last_hidden, pooled_output, feature_images):

        mean_hidden = torch.mean(last_hidden, dim = 1)

        x = torch.cat((mean_hidden, feature_images), dim=1)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.drop1(x)

        return self.linear2(x)


class ConcatBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(ConcatBERTModel, self).__init__()
        self.linear1 = nn.Linear(hyp_params.bert_hidden_size+hyp_params.image_feature_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear2 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, last_hidden, pooled_output, feature_images):

        x = torch.cat((pooled_output, feature_images), dim=1)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.drop1(x)

        return self.linear2(x)


class GatedAverageBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(GatedAverageBERTModel, self).__init__()
        self.gated_linear1 = GatedMultimodalLayer(hyp_params.bert_hidden_size, hyp_params.image_feature_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear1 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, last_hidden, pooled_output, feature_images):

        mean_hidden = torch.mean(last_hidden, dim = 1)

        x = self.gated_linear1(mean_hidden, feature_images)
        x = self.bn1(x)
        x = self.drop1(x)

        return self.linear1(x)
