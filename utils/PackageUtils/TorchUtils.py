import collections
import datetime
import io
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torchsummary

BatchContent = collections.namedtuple('BatchContent', 'inputs targets')


def summary(model, input_shape):
    old_stdout = sys.stdout

    new_stdout = io.StringIO()

    sys.stdout = new_stdout
    torchsummary.summary(model, input_shape)

    output = new_stdout.getvalue()

    sys.stdout = old_stdout

    return output


def get_torch_device():
    """
    Retrieves the device to run torch models, with preferability to GPU (denoted as cuda by torch)
    Returns: Device to run the models
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path):
    """
    Loads a Pytorch model
    Args:
        model_path: path to the model to load

    Returns: a model loaded from the specified path

    """
    logging.info(f"Load the model from: {model_path}")
    model = torch.load(model_path, map_location='cpu')
    logging.info(model)
    return model


def get_loader_shape(loader):
    assert len(loader) != 0
    return loader[0].shape


def split_batch_for_autoencoder(batch):
    """
    Creates a corresponding input-output match for AutoEncoder training
    :param batch: input data for AutoEncoder
    :return: tuple (inputs, targets)
            Since a model can have multiple inputs for different stems, the "inputs" variable is stored in a tuple
            where each entry is routed to its corresponding stem
    """
    return BatchContent(inputs=(batch,), targets=batch)


class TorchModel(nn.Module):
    """
    Wrapper class for a torch model to make it comfortable to train and load models
    """

    def __init__(self, model):
        super(TorchModel, self).__init__()
        self.iteration = 0
        self.model = model
        self.is_data_parallel = False
        self.evaluation_metrics = []
        self.callbacks = []

    def add_evaluation_metric(self, metric):
        self.evaluation_metrics.append(metric)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def register_callback(self, callback_fn):
        """
        Register a callback to be called after each evaluation run
        Args:
            callback_fn: a callable that accepts 2 inputs (output, target)
                            - output is the model's output
                            - target is the values of the target variable
        """
        self.callbacks.append(callback_fn)

    def data_parallel(self):
        """
        Transfers the model to data parallel mode
        """
        self.is_data_parallel = True
        if not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])

        return self

    @classmethod
    def load_model(cls, model_path):
        """
        Loads a pickled model
        Args:
            model_path: path to the pickled model

        Returns: TorchModel class instance wrapping the provided model
        """
        return cls(load_model(model_path))

    def notify_callbacks(self, notification, *args, **kwargs):
        for callback in self.callbacks:
            try:
                method = getattr(callback, notification)
                method(*args, **kwargs)
            except (AttributeError, TypeError) as e:
                logging.error(
                    f"callback {callback.__class__.__name__} doesn't fully implement the required interface {e}")

    def fit(self,
            train_iter,
            criterion,
            optimizer,
            eval_iter=None,
            test_iter=None,
            epochs=10,
            network_model_path_base=None,
            save_every=None,
            evaluate_every=None,
            batch_splitter=split_batch_for_autoencoder):
        """

        Args:
            train_iter: iterator for training
            criterion: loss function
            optimizer: optimizer for the algorithm
            eval_iter: iterator for evaluation
            test_iter: iterator for testing
            epochs: amount of epochs
            network_model_path_base: where to save the models
            save_every: saving model checkpoints every specified amount of epochs
            evaluate_every: perform evaluation every specified amount of epochs. If the evaluation is expensive,
                            you probably want ot choose a high value for this
            batch_splitter: function that splits a batch to inputs to the model and the targets.
                            For example, a batch_splitter for an autoencoder should copy the input to both model input and target:
                                model_input, model_target = batch, batch
                            In the general case, a batch contains both input variables and target variables, thus:
                                model_input, model_target = batch_splitter(batch)
        """
        criterion = criterion.to(self.device)
        self.notify_callbacks('on_training_start', epochs)

        for epoch in range(epochs):
            train_loss = self.do_epoch(criterion=criterion,
                                       optimizer=optimizer,
                                       data_iter=train_iter,
                                       epoch=epoch,
                                       batch_splitter=batch_splitter, )

            if save_every and network_model_path_base and epoch % save_every == 0:
                logging.info(f"Save the model after epoch {epoch}")
                self.save(os.path.join(network_model_path_base, f'epoch_{epoch}.pt'))

            val_loss = None
            if eval_iter and evaluate_every and epoch % evaluate_every == 0:
                logging.info(f"Evaluating after epoch {epoch}")
                val_loss = self.evaluate(criterion=criterion,
                                         data_iter=eval_iter,
                                         batch_splitter=batch_splitter, )

            test_loss = None
            if test_iter and evaluate_every and epoch % evaluate_every == 0:
                logging.info(f"Testing after epoch {epoch}")
                test_loss = self.evaluate(criterion=criterion,
                                          data_iter=test_iter,
                                          batch_splitter=batch_splitter, )

            self.notify_callbacks('on_training_iteration_end', train_loss, val_loss, test_loss)

        self.notify_callbacks('on_training_end', self.model)
        # Save the last model anyway...
        if network_model_path_base:
            self.save(os.path.join(network_model_path_base, f'epoch_{epoch + 1}.pt'))

    def evaluate(self, criterion, data_iter, batch_splitter=None):
        """
        Evaluates the model
        Args:
            criterion: Loss function for calculating the evaluation

            data_iter: torch data iterator

            batch_splitter: function that splits a batch to inputs to the model and the targets.
                            For example, a batch_splitter for an autoencoder should copy the input to both model input and target:
                                model_input, model_target = batch, batch
                            In the general case, a batch contains both input variables and target variables, thus:
                                model_input, model_target = batch_splitter(batch)

        """
        self.eval()
        self.notify_callbacks('on_evaluation_start', len(data_iter))
        total_loss = None
        all_targets = torch.tensor([])
        all_outputs = torch.tensor([])

        with torch.no_grad():
            for iteration, batch in enumerate(data_iter):
                # if iteration > 10:
                #     break
                if batch_splitter:
                    batch, targets = batch_splitter(batch)

                batch = self.data_to_device(batch, self.device)
                targets = self.data_to_device(targets, self.device)

                outputs = self.model(*batch)
                loss = criterion(outputs, targets)

                all_targets = torch.cat([all_targets, targets.detach().cpu()])
                all_outputs = torch.cat([all_outputs, outputs.detach().cpu()])

                self.notify_callbacks('on_evaluation_step',
                                      iteration,
                                      outputs.detach().cpu(),
                                      targets.detach().cpu(),
                                      [l.item() for l in loss] if type(loss) == tuple else loss.item())

                if total_loss is None:
                    if type(loss) == tuple:
                        total_loss = [l.item() for l in loss]
                    else:
                        total_loss = loss.item()
                else:
                    if type(loss) == tuple:
                        total_loss = [prev_l + new_l.item() for prev_l, new_l in zip(total_loss, loss)]
                    else:
                        total_loss += loss.item()

        if type(total_loss) == list:
            loss = [l / len(data_iter) for l in total_loss]
        else:
            loss = total_loss / len(data_iter)

        for metric in self.evaluation_metrics:
            logging.info(f"{metric.name}: {metric(all_targets, all_outputs)}")

        self.notify_callbacks('on_evaluation_end')
        return loss

    def do_epoch(self, criterion, optimizer, data_iter, epoch, batch_splitter=None):
        total_loss = None
        total_time = 0
        self.train()
        self.notify_callbacks('on_epoch_start', epoch, len(data_iter))
        for iteration, batch in enumerate(data_iter):
            self.iteration += 1
            start_time = time.time()
            if batch_splitter:
                batch, targets = batch_splitter(batch)

            batch = self.data_to_device(batch, self.device)
            targets = self.data_to_device(targets, self.device)

            outputs = self.model(*batch)

            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            if type(loss) == tuple:
                loss[0].backward()
            else:
                loss.backward()
            optimizer.step()

            if total_loss is None:
                if type(loss) == tuple:
                    total_loss = [l.item() for l in loss]
                else:
                    total_loss = loss.item()
            else:
                if type(loss) == tuple:
                    total_loss = [prev_l + new_l.item() for prev_l, new_l in zip(total_loss, loss)]
                else:
                    total_loss += loss.item()

            end_time = time.time()

            total_time += end_time - start_time

            self.notify_callbacks('on_epoch_step',
                                  self.iteration,
                                  iteration,
                                  [l.item() for l in loss] if type(loss) == tuple else loss.item(),
                                  )

            for metric in self.evaluation_metrics:
                logging.info(f"{metric.name}: {metric(targets.detach().cpu(), outputs.detach().cpu())}")

            self.iteration += 1

        if type(total_loss) == list:
            loss = [l / len(data_iter) for l in total_loss]
        else:
            loss = total_loss / len(data_iter)

        self.notify_callbacks('on_epoch_end', loss)
        return loss

    def data_to_device(self, data, device):
        """
        Transfers a tensor data to a device
        Args:
            data: torch tensor
            device: target device
        """
        if type(data) == list:
            data = [d.to(device) for d in data]
        elif type(data) == tuple:
            data = tuple([d.to(device) for d in data])
        else:
            data = data.to(device)

        return data

    def save(self, model_path):
        """
        Saves the model to the given path. If currently using data parallel, the method
        will save the original model and not the data parallel instance of it
        Args:
            model_path: target path to save the model to
        """
        if self.is_data_parallel:
            torch.save(self.model.module, model_path)
        else:
            torch.save(self.model, model_path)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
