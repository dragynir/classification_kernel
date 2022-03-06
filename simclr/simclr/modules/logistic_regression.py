import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes, projection=False, dropout=0.2):
        super(LogisticRegression, self).__init__()



        if projection:
            proj_features = n_features//2
            self.model = nn.Sequential(
                nn.Linear(n_features, proj_features),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_features, n_classes)
            )
        else:
            self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)
