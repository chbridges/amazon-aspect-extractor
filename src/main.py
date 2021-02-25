from torch.cuda import is_available
from utils.application import run_app
from pipeline import Pipeline

if __name__ == "__main__":
    device = "cuda" if is_available() else "cpu"

    P = Pipeline(
        output_size=3,
        model_name="laptops_best",
        device=device,
        normalize_output=False,
        n_layers=1,
        embedding_dim=300,
        hidden_dim=128,
        dropout=0.35,
        bidirectional=False,
    )
    run_app(P)
