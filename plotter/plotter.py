import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go


def parallel_coordinates(class_name: str = "val_loss.min"):
    df = pd.read_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v3.csv")

    resnet50_lstm_flickr8k = df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'LSTM') & (df['dataset.name'] == 'flickr8k')]
    resnet50_attention_flickr8k = df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'Attention') & (df['dataset.name'] == 'flickr8k')]

    lstm_columns_to_plot = ['num_layers', 'hidden_size', 'embed_size', 'encoder_dropout', 'dropout', 'encoder_lr', 'decoder_lr']
    match class_name:
        case "val_loss.min":
            lstm_columns_to_plot.append('val_loss.min')
        case "test_CIDEr.max":
            lstm_columns_to_plot.append('test_CIDEr.max')
        case "test_BLEU-4.max":
            lstm_columns_to_plot.append('test_BLEU-4.max')
        case _:
            raise ValueError(f"Invalid class name: {class_name}")
    resnet50_lstm_flickr8k = resnet50_lstm_flickr8k[lstm_columns_to_plot]

    lstm_labels = {
        "num_layers": "Num Layers",
        "hidden_size": "Hidden Size",
        "embed_size": "Embed Size",
        "encoder_dropout": "Encoder Dropout",
        "dropout": "Decoder Dropout",
        "encoder_lr": "Encoder LR",
        "decoder_lr": "Decoder LR",
        "val_loss.min": "Validation Loss",
        "test_CIDEr.max": "Test CIDEr",
        "test_BLEU-4.max": "Test BLEU-4"
    }
    fig = px.parallel_coordinates(resnet50_lstm_flickr8k, color=class_name, labels=lstm_labels,
                                  color_continuous_scale=plotly.colors.sequential.Plotly3)
    fig.show()
    # attention_values = {
    #     "Num Layers": [1, 2, 3],
    #     "Hidden Size": [256, 512, 1024],
    #     "Encoder Dropout": [0.1, 0.3, 0.5],
    #     "Decoder Dropout": [0.1, 0.3, 0.5],
    #     "Encoder LR": [1e-5, 5e-5, 0.0001, 0.0005],
    #     "Decoder LR": [1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005],
    #     "Attention Heads": [1, 2, 4, 8],
    # }
    fig2 = go.Figure(data=
                     go.Parcoords())


if __name__ == "__main__":
    parallel_coordinates('test_CIDEr.max')
