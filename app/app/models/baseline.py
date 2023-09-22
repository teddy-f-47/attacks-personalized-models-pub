from transformers import AutoModel
import torch.nn as nn

from app.settings import EMBEDDINGS_SIZES, TRANSFORMER_MODEL_STRINGS


class BaselineNet(nn.Module):
  def __init__(self, output_dim=2, bert_layer_model_name='distilbert', **kwargs):
    super(BaselineNet, self).__init__()
    self.bert_layer_model_name = bert_layer_model_name
    self.text_embedding_dim = EMBEDDINGS_SIZES[bert_layer_model_name]
    if bert_layer_model_name=='distilbert':
      self.pooler = nn.Linear(
        self.text_embedding_dim, self.text_embedding_dim
      )
      self.dropout = nn.Dropout(0.1)
    self.bert = AutoModel.from_pretrained(TRANSFORMER_MODEL_STRINGS[bert_layer_model_name])
    self.fc1 = nn.Linear(self.text_embedding_dim, output_dim)

  def forward(self, x_batch):
    input_ids = x_batch['input_ids']
    attention_mask = x_batch['attention_masks']
    if self.bert_layer_model_name=='distilbert':
      bert_output = self.bert(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=False
      )
      hidden_state = bert_output[0]
      pooled_output = hidden_state[:, 0]
      pooled_output = self.pooler(pooled_output)
      pooled_output = nn.ReLU()(pooled_output)
      pooled_output = self.dropout(pooled_output)
    else:
      _, pooled_output = self.bert(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=False
      )
    x = self.fc1(pooled_output)
    return x

  def freeze_bert(self):
    for name, param in self.bert.named_parameters():
      param.requires_grad = False

  def unfreeze_bert(self):
    for name, param in self.bert.named_parameters():
      param.requires_grad = True
