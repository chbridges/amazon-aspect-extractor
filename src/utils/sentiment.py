import os
from typing import List

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from utils.metrics import accuracy, log_sq_diff
from utils.preprocessing import review_to_int


class SentimentModel(nn.Module):
    """A sentiment predicting model
    Adapted from https://towardsdatascience.com/
                    sentiment-analysis-using-lstm-step-by-step-50d074f09948"""

    def __init__(
        self,
        dict_size: int,
        output_size: int = 1,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        n_layers: int = 10,
        normalize_output: bool = True,
        bidirectional: bool = True,
        dropout: float = 0.0,
        model_name: str = "LSTM",
        device: str = "cpu",
    ):
        """
        Arguments:
        - dict_size: number of tokens in the dataset
        - output_size: dimension of the output vector
        - embedding_dim: tokens are embedded into a vector of this dimension
        - hidden_dim: the hidden dimension of the lstm
        - n_layers: the number of layers for the lstm
        - normalize_output: whether to apply the sigmoid activation function
                            before returning the prediction
        - bidirectional: whether the lstm is evaluated in both directions
        - dropout: dropout chance of the lstm and before the final fully
                   connected layer
        - model_name: name to save the model under
        - device: name of device to run the model on (use gpu if available)"""

        super().__init__()
        self.name = model_name
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.device = device
        self.embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=0).to(
            self.device
        )

        # unused
        self.fc1 = nn.Linear(embedding_dim + 1, embedding_dim + 1).to(self.device)

        self.lstm = nn.LSTM(
            embedding_dim + 1,
            hidden_dim,
            n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        ).to(self.device)

        self.dropout = nn.Dropout(dropout).to(self.device)

        # Convolutional Model
        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim + 1, out_channels=32, kernel_size=3
        ).to(self.device)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3).to(
            self.device
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3).to(self.device)
        self.flatten = nn.Flatten().to(self.device)

        self.fc = nn.Linear(hidden_dim * (1 + self.bidirectional) + 32, output_size).to(
            self.device
        )
        self.normalize_output = normalize_output
        self.relu = nn.ReLU()
        if normalize_output:
            self.activation = nn.Sigmoid().to(self.device)
        else:
            self.activation = nn.Identity().to(self.device)

    def init_hidden(self, batch_size: int, stddev=1):
        """Reset the hidden state. Always call this before a new, unrelated prediction
        Arguments:
        - batch_size: size of the next incoming batch
        - stddev: the standard deviation of the gaussian used for initialization"""
        h = (
            torch.randn(
                self.n_layers * (1 + self.bidirectional),
                batch_size,
                self.hidden_dim,
            ).to(self.device)
            * stddev
        )
        c = (
            torch.randn(
                self.n_layers * (1 + self.bidirectional),
                batch_size,
                self.hidden_dim,
            ).to(self.device)
            * stddev
        )
        self.hidden = (h, c)

    def forward(self, x, s_lengths):
        """
        Arguments:
        - x: pytorch tensor with shape (batch_size, seq_len, 2)
        1st dim token numbers
        2nd a one hot encoding which tokens are considered keywords
        - s_lengths: length of the original unpadded reviews

        Returns:
        Sentiment prediction for the batch of inputs,
        shape (batch_size, self.output_size)"""

        batch_size, seq_len, _ = x.shape
        embedded = torch.cat(
            (self.embedding(x[:, :, 0]), x[:, :, 1].float().unsqueeze(-1)),
            dim=-1,
        )

        # unused
        # embedded = self.fc1(embedded)
        #
        # embedded = self.dropout(embedded)
        # embedded = self.relu(embedded)

        # Dynamic input packing adapted from
        # https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e

        # s_lengths, embedded = s_lengths[ind], embedded[ind]

        out = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, s_lengths, batch_first=True, enforce_sorted=False
        )

        out, self.hidden = self.lstm(out, self.hidden)

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = out.contiguous().view(batch_size, -1, out.shape[2])
        out, _ = torch.max(out, 1)
        out = self.dropout(out)

        out2 = self.conv1(embedded.permute(0, 2, 1))
        out2 = self.dropout(out2)
        out2 = self.relu(out2)
        out2 = self.max_pool(out2)
        out2 = self.conv2(out2)
        out2 = self.dropout(out2)
        out2 = self.relu(out2)
        out2, _ = torch.max(out2, -1)
        out2 = self.flatten(out2)

        out = self.fc(torch.cat((out, out2), dim=-1))

        out = self.activation(out)
        out = out.view(batch_size, self.output_size)
        # out[ind] = out

        return out


class SentimentDataset(Dataset):
    def __init__(self, reviews: List, aspects: List, sentiments: List):
        """
        Arguments:
        - reviews: a 2d list of shape (num_reviews, num_tokens)
        - aspects: a list of shape (num_reviews, num_tokens)
                   which contains a one-hot encoding for aspects
        - sentiments: a list of sentiment labels"""

        assert len(aspects) == len(
            reviews
        ), "Length of aspects and reviews doesnt match: {} - {}".format(
            len(aspects), len(reviews)
        )
        assert len(aspects) == len(
            sentiments
        ), "Length of aspects and sentiments doesnt match: {} - {}".format(
            len(aspects), len(sentiments)
        )

        if type(reviews[0][0]) == str:
            self.str_to_int_dict, self.int_to_str_dict = review_to_int(reviews)
            reviews = [
                [self.str_to_int_dict[token] for token in rev] for rev in reviews
            ]
        self.max_len = max([len(rev) for rev in reviews])
        self.dict_size = max([max(rev) for rev in reviews])
        print(
            "Reviews over 200 words: {}/{}".format(
                torch.sum(torch.BoolTensor([len(rev) > 200 for rev in reviews])),
                len(reviews),
            )
        )

        revs = [
            rev.copy() for i, rev in enumerate(reviews) for op in aspects[i]
        ]  # Duplicate reviews to have one per opinion
        asps = [
            op.copy() for i, asp in enumerate(aspects) for op in asp
        ]  # flatten opinions
        # flatten sentiments
        sentiments = [pol for sent in sentiments for pol in sent]
        self.seq_lens = [len(rev) for rev in revs]

        [asp.extend([0] * (self.max_len - len(asp))) for asp in asps]  # pad opinions
        [rev.extend([0] * (self.max_len - len(rev))) for rev in revs]  # pad reviews

        self.reviews = torch.Tensor(revs).type(torch.long)
        self.aspects = torch.Tensor(asps).type(torch.long)
        self.sentiments = torch.Tensor(sentiments).type(torch.float)
        assert self.reviews.shape == (
            len(revs),
            self.max_len,
        ), "Review tensor shape not recognized correctly by pytorch"
        assert self.aspects.shape == (
            len(asps),
            self.max_len,
        ), "Aspect tensor shape not recognized correctly by pytorch"

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx: int):
        return (
            torch.stack((self.reviews[idx], self.aspects[idx]), dim=-1),
            self.seq_lens[idx],
            self.sentiments[idx],
        )


def train_sentiment_model(
    model,
    optimizer,
    dataloaders,
    scheduler=None,
    criterion=log_sq_diff,
    n_epochs: int = 1,
    eval_every: int = 10,
    save_every: int = 10,
    save_best: bool = True,
    overwrite_chkpt: bool = True,
):
    """
    Train the sentiment LSTM on the dataloaders["train"] data

    Arguments:
    - model: the SentimentModel that is to be trained
    - optimizer: the optimizer used for training
    - dataloaders: a dictionary of pytorch dataloaders with keys train/val/test
    - criterion: the loss function to optimize
    - n_epochs: the number of epochs to train for
    - eval_every: in which epoch interval to evaluate on the val set
    - save_every: in which epoch interval to save the model
    - overwrite_chkpt: whether to overwrite saved models from earlier epochs"""
    best = -1

    for epoch in trange(n_epochs, desc="Epoch", leave=True, position=0):
        epoch_loss = 0
        for batch_id, (x, seq_lens, y) in tqdm(
            enumerate(dataloaders["train"]),
            desc="Batch train",
            leave=False,
            position=1,
        ):
            optimizer.zero_grad()
            model.init_hidden(len(x))

            x, seq_lens, y = (
                x.to(model.device),
                seq_lens.to(model.device),
                y.to(model.device),
            )
            pred = model(x, seq_lens)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / len(dataloaders["train"])

        tqdm.write("Epoch {} train loss: {}".format(epoch, epoch_loss))

        if not epoch % eval_every:
            val_loss = 0
            for batch_id, (x, seq_lens, y) in tqdm(
                enumerate(dataloaders["val"]),
                desc="Batch val",
                leave=False,
                position=1,
            ):
                with torch.no_grad():
                    model.init_hidden(len(x))

                    x, seq_lens, y = (
                        x.to(model.device),
                        seq_lens.to(model.device),
                        y.to(model.device),
                    )
                    pred = model(x, seq_lens)
                    loss = criterion(pred, y)
                    val_loss += loss.item() / len(dataloaders["val"])
            tqdm.write("Epoch {} validation loss: {}".format(epoch, val_loss))
            if best == -1 or val_loss < best:
                best = val_loss

                tqdm.write("Saving new best model...")
                save_name = os.path.join("models", model.name + "_best" + ".pth")

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "dict_for": model.dict_for,
                        "dict_back": model.dict_back,
                    },
                    save_name,
                )

        if type(scheduler) == ReduceLROnPlateau:
            scheduler.step(epoch_loss)
        elif scheduler is not None:
            scheduler.step()

        if not epoch % save_every:
            save_name = os.path.join("models", model.name + str(epoch) + ".pth")
            if overwrite_chkpt:
                save_name = os.path.join("models", model.name + ".pth")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dict_for": model.dict_for,
                    "dict_back": model.dict_back,
                },
                save_name,
            )

    # Save model after training
    save_name = os.path.join("models", model.name + str(n_epochs) + ".pth")
    if overwrite_chkpt:
        save_name = os.path.join("models", model.name + ".pth")

    torch.save(
        {
            "epoch": n_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dict_for": model.dict_for,
            "dict_back": model.dict_back,
        },
        save_name,
    )


def evaluate_sentiment_model(model, dataloaders, criterion=accuracy):
    """
    Evaluate the sentiment LSTM on the given criterion in batches
    on the train, val and test dataset
    Arguments:
    - model: the SentimentModel that should be evaluated
    - dataloaders: a dictionary of pytorch dataloaders with keys train/val/test
    - criterion: the function to evaluate the model on

    """
    acc = 0
    for batch_id, (x, seq_lens, y) in tqdm(
        enumerate(dataloaders["train"]), desc="Computing train accuracy"
    ):
        with torch.no_grad():
            model.init_hidden(len(x))

            x, seq_lens, y = (
                x.to(model.device),
                seq_lens.to(model.device),
                y.to(model.device),
            )
            pred = model(x, seq_lens)
            acc += criterion(pred, y, regression=(model.output_size == 1)) / len(
                dataloaders["train"]
            )
    print(
        "Training results of model {} on criterion {}: {}".format(
            model.name, criterion.__name__, acc
        )
    )
    acc = 0
    for batch_id, (x, seq_lens, y) in tqdm(
        enumerate(dataloaders["val"]), desc="Computing validation accuracy"
    ):
        with torch.no_grad():
            model.init_hidden(len(x))

            x, seq_lens, y = (
                x.to(model.device),
                seq_lens.to(model.device),
                y.to(model.device),
            )
            pred = model(x, seq_lens)
            acc += criterion(pred, y, regression=(model.output_size == 1)) / len(
                dataloaders["val"]
            )
    print(
        "Validation results of model {} on criterion {}: {}".format(
            model.name, criterion.__name__, acc
        )
    )
    acc = 0
    for batch_id, (x, seq_lens, y) in tqdm(
        enumerate(dataloaders["test"]), desc="Computing test accuracy"
    ):
        with torch.no_grad():
            model.init_hidden(len(x))

            x, seq_lens, y = (
                x.to(model.device),
                seq_lens.to(model.device),
                y.to(model.device),
            )
            pred = model(x, seq_lens)
            acc += criterion(pred, y, regression=(model.output_size == 1)) / len(
                dataloaders["val"]
            )
    print(
        "Test results of model {} on criterion {}: {}".format(
            model.name, criterion.__name__, acc
        )
    )
