class AccuracyTopK:
    def __init__(self, k=1):
        self.name = f'accuracy_top_{k}'
        self.k = k

    def __call__(self, y_true, y_pred):
        topk_indices = y_pred.topk(self.k).indices
        n_true = (topk_indices == y_true.view(-1, 1).expand_as(topk_indices)).sum()
        return n_true / len(y_pred)
