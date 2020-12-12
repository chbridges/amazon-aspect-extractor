import os

from utils.keywords import extract_keyword_list
from utils.dataloading import load_amazon_mulitlingual, load_semeval2015
from utils.preprocessing import PreprocessingPipeline
from utils.sentiment import SentimentModel, SentimentDataset, train_sentiment_model, evaluate_sentiment_model
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda import is_available

if __name__ == "__main__":
    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/amazon_multilingual") #path to dataset
    # dataset = load_amazon_mulitlingual(path, select_languages=["en"])
    # reviewtext = [rev["review_body"] for rev in dataset["train"]]
    # keywords = extract_keyword_list(reviewtext, minScore=2.0)
    # print(keywords[:10])
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/ABSA15")
    dataset = load_semeval2015(path, categories=["restaurants"])
    pipeline = PreprocessingPipeline()
    dataloaders = {}
    str_to_int_dict, int_to_str_dict = {}, {}

    for phase in dataset.keys():
        revtexts, aspects, sentiment, dict_for, dict_back = pipeline(dataset[phase])
        #TODO Combine the dictionaries for the vocab
        sentimentds = SentimentDataset(revtexts, aspects, sentiment)
        dataloaders[phase] = DataLoader(sentimentds, batch_size=32, shuffle=True)

    device = "cuda" if is_available() else "cpu"
    model = SentimentModel(len(str_to_int_dict)+1, model_name="LSTM_run1", device=device)
    optimizer = Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer)
    train_sentiment_model(model, optimizer, dataloaders, scheduler=scheduler, n_epochs=10, eval_every=1)
    evaluate_sentiment_model(model)
