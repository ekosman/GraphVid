from sklearn.metrics import accuracy_score


class Accuracy:
    def __init__(self):
        self.name = 'accuracy'

    def __call__(self, y_true, y_pred):
        y_pred = y_pred.argmax(dim=-1, keepdim=True).view(-1)
        return accuracy_score(y_true, y_pred)
