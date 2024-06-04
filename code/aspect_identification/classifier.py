from torch import nn
from transformers import AutoModel


class Classifier(nn.Module):
    def __init__(self, pre_trained_model, label_num):
        super(Classifier, self).__init__()

        self.pretrained_model = AutoModel.from_pretrained(pre_trained_model)
        hidden_size = self.pretrained_model.config.hidden_size

        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, label_num)
        )

    def forward(self, input_ids, attention_mask):
        embeddings = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_embedding = embeddings[:, 0, :]
        logits = self.cls(cls_embedding)
        return logits
