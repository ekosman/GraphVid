from sklearn.metrics import average_precision_score
import torch


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


class mAP:
    def __init__(self, tb_writer=None):
        self.name = f'mAP'
        self.iteration = 0
        self.tb_writer = tb_writer

    def __call__(self, y_true, y_pred, phase):
        idx = (y_true.sum(dim=1) != 0)
        y_true, y_pred = y_true[idx], y_pred[idx]

        precisions = torch.zeros(y_true.shape[1])
        for i, class_ in enumerate(range(y_true.shape[1])):
            y_true_tmp = y_true[:, class_]
            y_pred_tmp = y_pred[:, class_]

            n_positives = y_true_tmp.sum()
            AP = average_precision_score(y_true_tmp, y_pred_tmp) if n_positives != 0 else 0
            precisions[i] = AP

        mAP = precisions.mean()

        if self.tb_writer is not None:
            self.tb_writer.add_scalar(f'{self.name}/{phase}', mAP, self.iteration)
            self.iteration += 1

        return mAP