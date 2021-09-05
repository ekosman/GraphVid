class AccuracyTopK:
    def __init__(self, k=1, tb_writer=None):
        self.name = f'accuracy_top_{k}'
        self.k = k
        self.iteration = 0
        self.tb_writer = tb_writer

    def __call__(self, y_true, y_pred, phase):
        topk_indices = y_pred.topk(self.k).indices
        n_true = (topk_indices == y_true.view(-1, 1).expand_as(topk_indices)).sum()
        res = n_true / len(y_pred)

        if self.tb_writer is not None:
            self.tb_writer.add_scalar(f'{self.name}/{phase}', res, self.iteration)
            self.iteration += 1

        return res
