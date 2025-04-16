import pandas as pd
import plotly
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def parallel_coordinates(class_name: str = "val_loss.min", model_name: str = "ResNet50-LSTM"):
    df = pd.read_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v3.csv")

    columns_to_plot = ['num_layers', 'hidden_size', 'encoder_dropout', 'dropout', 'encoder_lr', 'decoder_lr']
    match model_name:
        case "ResNet50-LSTM":
            model_df = df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'LSTM') & (df['dataset.name'] == 'flickr8k')]
            columns_to_plot.append('embed_size')
        case "ResNet50-Attention":
            model_df = df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'Attention') & (df['dataset.name'] == 'flickr8k')]
            columns_to_plot.append('num_heads')
        case _:
            raise ValueError(f"Invalid model name: {model_name}")

    match class_name:
        case "val_loss.min":
            columns_to_plot.append('val_loss.min')
        case "test_CIDEr.max":
            columns_to_plot.append('test_CIDEr.max')
        case "test_BLEU-4.max":
            columns_to_plot.append('test_BLEU-4.max')
        case _:
            raise ValueError(f"Invalid class name: {class_name}")

    model_df = model_df[columns_to_plot]

    labels = {
        "num_layers": "Num Layers",
        "hidden_size": "Hidden Size",
        "embed_size": "Embed Size",
        "encoder_dropout": "Encoder Dropout",
        "dropout": "Decoder Dropout",
        "encoder_lr": "Encoder LR",
        "decoder_lr": "Decoder LR",
        "num_heads": "Num Heads",
        "val_loss.min": "Validation Loss",
        "test_CIDEr.max": "Test CIDEr",
        "test_BLEU-4.max": "Test BLEU-4"
    }
    fig = px.parallel_coordinates(model_df, color=class_name, labels=labels, color_continuous_scale=plotly.colors.sequential.Plotly3)
    fig.show()


def correlation_importance(class_name: str = "val_loss.min", model_name: str = "ResNet50-LSTM"):
    df = pd.read_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v3.csv")

    columns_to_analyze = ['num_layers', 'hidden_size', 'encoder_dropout', 'dropout', 'encoder_lr', 'decoder_lr']
    match model_name:
        case "ResNet50-LSTM":
            model_df = df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'LSTM') & (df['dataset.name'] == 'flickr8k')]
            columns_to_analyze.append('embed_size')
        case "ResNet50-Attention":
            model_df = df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'Attention') & (df['dataset.name'] == 'flickr8k')]
            columns_to_analyze.append('num_heads')
        case _:
            raise ValueError(f"Invalid model name: {model_name}")

    match class_name:
        case "val_loss.min":
            columns_to_analyze.append('val_loss.min')
        case "test_CIDEr.max":
            columns_to_analyze.append('test_CIDEr.max')
        case "test_BLEU-4.max":
            columns_to_analyze.append('test_BLEU-4.max')
        case _:
            raise ValueError(f"Invalid class name: {class_name}")

    model_df = model_df[columns_to_analyze]

    labels = {
        "num_layers": "Num Layers",
        "hidden_size": "Hidden Size",
        "embed_size": "Embed Size",
        "encoder_dropout": "Encoder Dropout",
        "dropout": "Decoder Dropout",
        "encoder_lr": "Encoder LR",
        "decoder_lr": "Decoder LR",
        "num_heads": "Num Heads",
        "val_loss.min": "Validation Loss",
        "test_CIDEr.max": "Test CIDEr",
        "test_BLEU-4.max": "Test BLEU-4"
    }

    # Calculate correlations
    correlation = model_df.corr()[class_name].drop(class_name)
    correlation.index = correlation.index.map(labels)

    print(f"{labels[class_name]} Correlation and Permutation Importance in {model_name}")
    print("Correlation with Target:")
    print(correlation.sort_values(ascending=False).to_string(float_format="%.3f"))

    # Calculate permutation importance
    features = model_df.drop(columns=[class_name])
    target = model_df[class_name]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)

    result = permutation_importance(model, features, target, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': result.importances_mean
    }).sort_values('Importance', ascending=False)
    importance_df['Feature'] = importance_df['Feature'].map(labels)

    print("\nPermutation Importance:")
    print(importance_df.to_string(index=False, float_format="%.3f"))

    diverging_palette = sns.diverging_palette(h_neg=260, h_pos=140, s=80, l=60, n=7, sep=1, center="light")

    plt.figure(figsize=(9, 3))
    plt.suptitle(f"{labels[class_name]} Correlation and Permutation Importance in {model_name}")
    plt.subplot(1, 2, 1)
    ax1 = sns.barplot(x=correlation.values, y=correlation.index, hue=correlation.values, legend=False, palette=diverging_palette)
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.3f", label_type="edge", padding=3, fontsize=9)
    plt.title(f'Correlation with {labels[class_name]}')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Features')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlim(-1, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.5, color='#cccccc')  # X-axis grid

    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Importance', legend=False, palette=diverging_palette)
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.3f", label_type="edge", padding=3, fontsize=9)
    plt.title('Permutation Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.xlim(0, 1.8)
    plt.grid(axis='x', linestyle='--', alpha=0.5, color='#cccccc')  # X-axis grid
    plt.tight_layout()
    plt.savefig(f"../results/{model_name}_{class_name}_correlation_importance.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_validation_loss(model_name: str = "ResNet50-LSTM", dataset_name: str = "flickr8k"):
    # Read CSV
    val_loss_df = pd.read_csv("../plots/results/csv_source/val_loss_v4.csv")
    experiments_df = pd.read_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v3.csv")

    match model_name:
        case "ResNet50-LSTM":
            model_df = experiments_df[
                (experiments_df['encoder'] == 'resnet50') & (experiments_df['decoder'] == 'LSTM') & (experiments_df['dataset.name'] == dataset_name)]
        case "ResNet50-Attention":
            model_df = experiments_df[(experiments_df['encoder'] == 'resnet50') & (experiments_df['decoder'] == 'Attention') & (
                    experiments_df['dataset.name'] == dataset_name)]
        case "Swin-LSTM":
            model_df = experiments_df[
                (experiments_df['encoder'] == 'swin') & (experiments_df['decoder'] == 'LSTM') & (experiments_df['dataset.name'] == dataset_name)]
        case "Swin-Attention":
            model_df = experiments_df[
                (experiments_df['encoder'] == 'swin') & (experiments_df['decoder'] == 'Attention') & (experiments_df['dataset.name'] == dataset_name)]
        case _:
            raise ValueError(f"Invalid model name: {model_name}")

    # Filter validation loss columns

    experiments_names = model_df['Name'].unique()
    val_loss_filter = [col for col in val_loss_df.columns if col in experiments_names]
    print(val_loss_filter)  # empty list? help??
    filtered_val_loss_df = val_loss_df[val_loss_filter]

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot each validation loss curve
    for col in filtered_val_loss_df.columns:
        plt.plot(filtered_val_loss_df.index + 1, filtered_val_loss_df[col], label=col)

    # Add labels and title
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(f"Validation Loss Comparison")
    plt.ylim(0, 5)
    plt.grid(True, alpha=0.3)

    # Show plot
    plt.tight_layout()
    plt.show()


# Execute the function

if __name__ == "__main__":
    plot_validation_loss("Swin-Attention")
    # correlation_importance('val_loss.min', 'ResNet50-Attention')
    # correlation_importance('test_CIDEr.max', 'ResNet50-Attention')
    # correlation_importance('test_BLEU-4.max', 'ResNet50-Attention')
    # correlation_importance('val_loss.min', 'ResNet50-LSTM')
    # correlation_importance('test_CIDEr.max', 'ResNet50-LSTM')
    # correlation_importance('test_BLEU-4.max', 'ResNet50-LSTM')
