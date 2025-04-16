import pandas as pd
import plotly
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def parallel_coordinates(class_name: str = "val_loss.min", model_name: str = "ResNet50-LSTM"):
    labels, experiments = load_and_filter_correlation(class_name, model_name)
    fig = px.parallel_coordinates(experiments, color=class_name, labels=labels, color_continuous_scale=plotly.colors.sequential.Plotly3)
    fig.show()


def correlation_importance(class_name: str = "val_loss.min", model_name: str = "ResNet50-LSTM"):
    labels, experiments = load_and_filter_correlation(class_name, model_name)

    # Calculate correlations
    correlation = experiments.corr()[class_name].drop(class_name)
    correlation.index = correlation.index.map(labels)

    print(f"{labels[class_name]} Correlation and Permutation Importance in {model_name}")
    print("Correlation with Target:")
    print(correlation.sort_values(ascending=False).to_string(float_format="%.3f"))

    # Calculate permutation importance
    features = experiments.drop(columns=[class_name])
    target = experiments[class_name]

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


def plot_validation_loss(model_name: str, dataset_name: str, min_y=2, max_y=6):
    val_loss = pd.read_csv("../plots/results/csv_source/val_loss_v4.csv")
    experiments_filtered = filter_experiments(dataset_name, model_name)
    experiments_names = experiments_filtered['Name'].unique()
    val_loss_filter = [col for col in val_loss.columns if col in experiments_names]
    val_loss_filtered = val_loss[val_loss_filter]

    plt.figure(figsize=(15, 10))

    # Plot each validation loss curve
    for col in val_loss_filtered.columns:
        plt.plot(val_loss_filtered.index + 1, val_loss_filtered[col], label=col)

    # Horizontal line for minimum loss
    min_loss = val_loss_filtered.min().min()  # Find overall minimum loss
    plt.axhline(y=min_loss, color='0', linestyle='--', label=f'Minimum Loss ({min_loss:.2f})')

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    if model_name is None and dataset_name is None:
        plt.title("Validation Loss Comparison")
    elif dataset_name is None:
        plt.title(f"Validation Loss Comparison for {model_name}")
    elif model_name is None:
        plt.title(f"Validation Loss Comparison for {dataset_name}")
    else:
        plt.title(f"Validation Loss Comparison for {model_name} on {dataset_name}")
    plt.ylim(min_y, max_y)
    plt.grid(True, alpha=0.3)
    plt.text(x=0.99, y=min_loss + 0.02, s=f'Min: {min_loss:.2f}', ha='right', va='bottom', color='0', transform=plt.gca().get_yaxis_transform())
    # Show plot
    plt.tight_layout()
    plt.savefig(f"../plots/results/val_loss_{model_name}_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


def create_boxplot(metric: str, model_names: list[str], dataset_name: str, min_y: float = 2.3, max_y: float = 5):
    experiments_filtered = []
    # experiments_filtered = filter_experiments(dataset_name, model_names)
    for model_name in model_names:
        experiments = filter_experiments(dataset_name, model_name)
        experiments_filtered.append(experiments)
    # Create the boxplot
    plt.figure(figsize=(10, 10))
    label = {
        "num_heads": "Num Heads",
        "val_loss.min": "Validation Loss",
        "test_CIDEr.max": "Test CIDEr",
        "test_BLEU-4.max": "Test BLEU-4",
        "flickr8k": "Flickr8k",
        "coco": "COCO",
    }
    for i, model_name in enumerate(model_names):
        plt.subplot(1, len(model_names), i + 1)
        sns.boxplot(data=experiments_filtered[i], y=metric, color="cornflowerblue")

        # Calculate statistics
        mean_val = experiments_filtered[i][metric].mean()
        min_val = experiments_filtered[i][metric].min()
        max_val = experiments_filtered[i][metric].max()

        # Add text annotation
        stats_text = f"Mean: {mean_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}"
        plt.text(0.25, 4.9, stats_text, verticalalignment='top', horizontalalignment='left', color='black', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='0', boxstyle='round,pad=0.65'))
        plt.xlabel(f"Model {model_name}")
        plt.ylabel(label[metric] if i == 0 else "")
        plt.ylim(min_y, max_y)
        plt.grid(True, alpha=0.3)
    plt.suptitle(f"Boxplot of {label[metric]} on {label[dataset_name]}")
    plt.tight_layout(pad=0.5, rect=(0, 0, 0.98, 0.98))
    # plt.savefig(f"../plots/results/boxplot_{metric}_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.show()


def load_and_filter_correlation(class_name, model_name):
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
    return labels, model_df


def filter_experiments(dataset_name, model_name):
    experiments = pd.read_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v3.csv")
    experiments_filtered = experiments
    match model_name:
        case "ResNet50-LSTM":
            experiments_filtered = experiments[
                (experiments['encoder'] == 'resnet50') & (experiments['decoder'] == 'LSTM')]
        case "ResNet50-Attention":
            experiments_filtered = experiments[(experiments['encoder'] == 'resnet50') & (experiments['decoder'] == 'Attention') & (
                    experiments['dataset.name'] == dataset_name)]
        case "Swin-Attention":
            experiments_filtered = experiments[
                (experiments['encoder'] == 'swin') & (experiments['decoder'] == 'Attention')]
    match dataset_name:
        case "flickr8k":
            experiments_filtered = experiments_filtered[experiments_filtered['dataset.name'] == 'flickr8k']
        case "coco":
            experiments_filtered = experiments_filtered[experiments_filtered['dataset.name'] == 'coco']
    return experiments_filtered


if __name__ == "__main__":
    create_boxplot("val_loss.min", ["ResNet50-LSTM", "ResNet50-Attention"], "flickr8k")
    create_boxplot("test_CIDEr.max", ["ResNet50-LSTM", "ResNet50-Attention"], "flickr8k", -0.01, 0.5)
    create_boxplot("test_BLEU-4.max", ["ResNet50-LSTM", "ResNet50-Attention"], "flickr8k", -0.01, 0.2)
