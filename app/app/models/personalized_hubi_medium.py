from transformers import AutoModel
import torch.nn as nn

from app.settings import EMBEDDINGS_SIZES, TRANSFORMER_MODEL_STRINGS


class PerHuBiMedNet(nn.Module):
  def __init__(
    self, word_num, annotator_num, output_dim=2, bert_layer_model_name='distilbert',
    classifier_embedding_dim=20, hidden_dim=100, dp_annotator_embeddings=0.2,
    **kwargs
  ):
    super(PerHuBiMedNet, self).__init__()
    self.bert_layer_model_name = bert_layer_model_name
    self.text_embedding_dim = EMBEDDINGS_SIZES[bert_layer_model_name]

    self.annotator_embeddings = nn.Embedding(
      num_embeddings=annotator_num,
      embedding_dim=classifier_embedding_dim,
      padding_idx=0
    )
    self.word_biases = nn.Embedding(
      num_embeddings=word_num,
      embedding_dim=output_dim,
      padding_idx=0
    )
    self.annotator_embeddings.weight.data.uniform_(-.01, .01)
    self.word_biases.weight.data.uniform_(-.01, .01)

    if bert_layer_model_name=='distilbert':
      self.pooler = nn.Linear(
        self.text_embedding_dim, self.text_embedding_dim
      )
      self.dropout = nn.Dropout(0.1)

    self.bert = AutoModel.from_pretrained(TRANSFORMER_MODEL_STRINGS[bert_layer_model_name])
    self.softplus = nn.Softplus()
    self.fc1 = nn.Linear(self.text_embedding_dim, hidden_dim)
    self.dp_ann_emb = nn.Dropout(p=dp_annotator_embeddings)
    self.fc_annotator = nn.Linear(classifier_embedding_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, output_dim)


  def forward(self, x_batch):
    input_ids = x_batch['input_ids']
    attention_mask = x_batch['attention_masks']
    annotator_ids = x_batch['annotator_ids']
    tokens = x_batch['tokens_sorted']

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

    x = pooled_output.view(-1, self.text_embedding_dim)
    x = self.fc1(x)
    x = self.softplus(x)

    annotator_embeddings = self.annotator_embeddings(annotator_ids)
    annotator_embeddings = self.dp_ann_emb(annotator_embeddings)
    annotator_embeddings = self.fc_annotator(annotator_embeddings)
    annotator_embeddings = self.softplus(annotator_embeddings)

    word_biases = self.word_biases(tokens)
    word_biases_mask = tokens != 0
    word_biases = (word_biases*word_biases_mask[:, :, None]).sum(dim=1)

    x = self.fc2(x * annotator_embeddings) + word_biases

    return x

  def freeze_bert(self):
    for name, param in self.bert.named_parameters():
      param.requires_grad = False

  def unfreeze_bert(self):
    for name, param in self.bert.named_parameters():
      param.requires_grad = True
