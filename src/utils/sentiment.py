import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm, trange
from utils.preprocessing import review_to_int
import os

class SentimentModel(nn.Module):
    """A sentiment predicting model"""

    def __init__(self,
                 dict_size: int,
                 output_size: int = 1,
                 embedding_dim: int = 512,
                 hidden_dim: int = 256,
                 n_layers: int = 10,
                 normalize_output: bool = True,
                 bidirectional: bool = True,
                 dropout: float = 0.0,
                 model_name: str = "LSTM",
                 device="cpu"):
        """
        Arguments:
        - dict_size: number of tokens in the dataset
        - output_size: dimension of the output vector
        - embedding_dim: tokens are embedded into a vector of this dimension
        - hidden_dim: the hidden dimension of the lstm
        - n_layers: the number of layers for the lstm
        - normalize_output: whether to apply the sigmoid activation function before returning the prediction
        - bidirectional: whether the lstm is evaluated in both directions
        - dropout: dropout chance of the lstm and before the final fully connected layer
        - model_name: name to save the model under
        - device: name of device to run the model on (use gpu if available)"""

        super().__init__()

        self.name = model_name
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.device = device
        self.embedding = nn.Embedding(dict_size, embedding_dim, padding_idx = 0).to(self.device)
        self.lstm = nn.LSTM(embedding_dim + 1,
                            hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)
        self.fc = nn.Linear(hidden_dim*(1+self.bidirectional), output_size).to(self.device)
        self.normalize_output = normalize_output
        if normalize_output:
            self.activation = nn.Sigmoid().to(self.device)
        else:
            self.activation = nn.Identity().to(self.device)


    def init_hidden(self, batch_size: int):
        """Reset the hidden state. Always call this before a new, unrelated prediction
        Arguments:
        - batch_size: size of the next incoming batch"""
        h = torch.randn(self.n_layers*(1+self.bidirectional), batch_size, self.hidden_dim).to(self.device)
        c = torch.randn(self.n_layers*(1+self.bidirectional), batch_size, self.hidden_dim).to(self.device)
        self.hidden = (h, c)

    def forward(self, x, s_lengths):
        """
        Arguments:
        - x: pytorch tensor with shape (batch_size, seq_len, 2)
        1st dim token numbers
        2nd a one hot encoding which tokens are considered keywords
        - s_lengths: length of the original unpadded reviews

        Returns:
        Sentiment prediction for the batch of inputs, shape (batch_size, seq_len, self.output_size)"""

        batch_size, seq_len, _ = x.shape
        embedded = torch.cat((self.embedding(x[:, :, 0]), x[:, :, 1].float().unsqueeze(-1)), dim=-1)


        out = torch.nn.utils.rnn.pack_padded_sequence(embedded, s_lengths, batch_first=True)

        out, self.hidden = self.lstm(out, self.hidden)

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = out.contiguous().view(-1, out.shape[2])

        out = self.dropout(out)
        out = self.fc(out)
        out = self.activation(out)

        return out.view(batch_size, seq_len, self.output_size)[:,-1,:]


class SentimentDataset(Dataset):

    def __init__(self, reviews, aspects, sentiments):
        """
        Arguments:
        - reviews: a 2d list of shape (num_reviews, num_tokens)
        - aspects: a list of shape (num_reviews, num_tokens), which contains a one-hot encoding for aspects
        - sentiments: a list of sentiment labels"""

        assert len(aspects) == len(reviews), "Length of aspects and reviews doesnt match: {} - {}".format(len(aspects), len(reviews))
        assert len(aspects) == len(sentiments), "Length of aspects and sentiments doesnt match: {} - {}".format(len(aspects), len(sentiments))

        if type(reviews[0][0]) == str:
            self.str_to_int_dict, self.int_to_str_dict = review_to_int(reviews)
            reviews = [[self.str_to_int_dict[token] for token in rev] for rev in reviews]

        self.max_len = max([len(rev) for rev in reviews])
        self.dict_size = max([max(rev) for rev in reviews])


        reviews = [rev for i, rev in enumerate(reviews) for op in aspects[i]] #Duplicate reviews to have one per opinion
        aspects = [op for asp in aspects for op in asp] # flatten opinions
        sentiments = [pol for sent in sentiments for pol in sent] #flatten sentiments


        [asp.extend([0] * (self.max_len - len(asp))) for asp in aspects] #pad opinions
        [rev.extend([0] * (self.max_len - len(rev))) for rev in reviews] #pad reviews

        self.seq_lens = [len(rev) for rev in reviews]

        self.reviews = torch.Tensor(reviews).type(torch.long)
        self.aspects = torch.Tensor(aspects).type(torch.long)
        self.sentiments = torch.Tensor(sentiments).type(torch.float)

        assert self.reviews.shape == (len(reviews), self.max_len), "Review tensor shape not recognized correctly by pytorch"
        assert self.aspects.shape == (len(aspects), self.max_len), "Aspect tensor shape not recognized correctly by pytorch"

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return torch.stack((self.reviews[idx], self.aspects[idx]), dim=-1), self.seq_lens[idx], self.sentiments[idx]


def log_sq_diff(pred, y):
    """Logarithmic squared difference loss.
    This ranges from 0 to -infinity and is subject to minimization
    Arguments:
    - pred: a vector of sentiment predictions between 0 and 1
    - y: a vector of sentiment ground truths between 0 and 1

    Returns:
    logarithmic squared difference of the two vectors"""
    return torch.log(torch.mean((pred - y)**2))

def accuracy(pred, y):
    """Accuracy of a prediction pred vs. ground truth y
    Arguments:
    - pred: a vector of sentiment predictions between 0 and 1
    - y: a vector of sentiment ground truths between 0 and 1

    Returns:
    Accuracy of the predictions when rounded to the nearest label (0/0.5/1)"""
    labels = torch.round(pred*2).type(torch.int)
    y = (y*2).type(torch.int)
    return torch.mean((labels == y).float())

def train_sentiment_model(model, optimizer, dataloaders, scheduler=None, criterion=log_sq_diff, n_epochs=1, eval_every=10, save_every=10, overwrite_chkpt=True):
    """
    Arguments:
    - model: the SentimentModel that is to be trained
    - optimizer: the optimizer used for training
    - dataloaders: a dictionary of pytorch dataloaders with keys train/val/(test)
    - criterion: the loss function to optimize
    - n_epochs: the number of epochs to train for
    - eval_every: in which epoch interval to evaluate on the val set
    - save_every: in which epoch interval to save the model
    - overwrite_chkpt: whether to overwrite saved models from earlier epochs"""

    for epoch in trange(n_epochs, desc='Epoch', leave=True, position=0):
        epoch_loss = 0
        for batch_id, (x, seq_lens, y) in tqdm(enumerate(dataloaders["train"]), desc=f'Batch train', leave=False, position=1):
            optimizer.zero_grad()
            model.init_hidden(len(x))

            x, seq_lens, y = x.to(model.device), seq_lens.to(model.device), y.to(model.device)
            pred = model(x, seq_lens)
            if not model.normalize_output:
                pred = nn.Sigmoid()(pred)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()/len(dataloaders["train"])

        tqdm.write("Epoch {} train loss: {}".format(epoch, epoch_loss))

        if not epoch % eval_every:
            val_loss = 0
            for batch_id, (x, seq_lens, y) in tqdm(enumerate(dataloaders["val"]), desc=f'Batch val', leave=False, position=1):
                with torch.no_grad():
                    model.init_hidden(len(x))

                    x, seq_lens, y = x.to(model.device), seq_lens.to(model.device), y.to(model.device)
                    pred = model(x, seq_lens)
                    if not model.normalize_output:
                        pred = nn.Sigmoid()(pred)
                    loss = criterion(pred, y)
                    val_loss += loss.item()/len(dataloaders["val"])
            tqdm.write("Epoch {} validation loss: {}".format(epoch, val_loss))

            if not scheduler is None:
                scheduler.step(val_loss)

        if not epoch % save_every:
            save_name = os.path.join("models", model.name + str(epoch) + ".pth")
            if overwrite_chkpt:
                save_name = os.path.join("models", model.name + ".pth")

            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                        save_name)

    #Save model after training
    save_name = os.path.join("models", model.name + str(n_epochs) + ".pth")
    if overwrite_chkpt:
        save_name = os.path.join("models", model.name + ".pth")

    torch.save({"epoch": n_epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
                save_name)

def evaluate_sentiment_model(model, dataloaders, criterion=accuracy):
    acc = 0
    for batch_id, (x, seq_lens, y) in tqdm(enumerate(dataloaders["val"]), desc=f'Computing accuracy'):
        with torch.no_grad():
            model.init_hidden(len(x))

            x, seq_lens, y = x.to(model.device), seq_lens.to(model.device), y.to(model.device)
            pred = model(x, seq_lens)
            if not model.normalize_output:
                pred = nn.Sigmoid()(pred)
            acc += criterion(pred, y)/len(dataloaders["val"])
    print("Validation results of model {} on criterion {}: {}".format(model.name, criterion.__name__, acc))
