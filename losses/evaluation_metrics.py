class AccuracyTopK:
    def __init__(self, k=1):
        self.name = f'accuracy_top_{k}'
        self.k = k

    def __call__(self, y_true, y_pred):
        topk_indices = y_pred.topk(self.k).indices
        n_true = (topk_indices == y_true).sum()
        return n_true / len(y_pred)
