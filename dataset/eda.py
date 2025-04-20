# EDA of Flickr8k Dataset and Coco Dataset for Image Captioning
import string
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords


def caption_len(train_captions: pd.DataFrame, val_captions: pd.DataFrame, test_captions: pd.DataFrame, save_name: str):
    """
    Analyzes and visualizes caption lengths across dataset splits.
    :param train_captions: DataFrame with 'caption' column
    :param val_captions: DataFrame with 'caption' column
    :param test_captions: DataFrame with 'caption' column
    :param save_name: Name for saving plots
    :returns: Dictionary of length statistics for each split
    """

    splits = {
        'Train': train_captions,
        'Validation': val_captions,
        'Test': test_captions
    }

    length_data = {}
    stats = {}

    # Process each split
    for split_name, df in splits.items():
        if df is not None:
            # Tokenize and calculate lengths
            lengths = df['caption'].apply(lambda x: len(nltk.word_tokenize(x))).tolist()
            length_data[split_name] = lengths

            # Calculate statistics
            stats[split_name] = {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'count': len(df)
            }

    # Plotting
    # Histogram
    plt.figure(figsize=(10, 6))
    for split_name, lengths in length_data.items():
        plt.hist(lengths, bins=30, alpha=0.6, label=split_name,
                 color=("purple", "firebrick", "gold")[list(splits.keys()).index(split_name)])
    plt.xlabel('Caption Length (words)')
    plt.ylabel('Frequency')
    plt.title('Caption Length Distribution')
    plt.legend()
    plt.tight_layout()
    if save_name:
        plt.savefig("../plots/eda/histogram_" + save_name + ".png", dpi=400, bbox_inches='tight')
    plt.show()

    # Boxplot
    plt.figure(figsize=(10, 10))
    plt.boxplot(length_data.values(), tick_labels=length_data.keys())
    plt.ylabel('Caption Length (words)')
    plt.title('Caption Lengths Across Splits')
    plt.tight_layout()
    if save_name:
        plt.savefig("../plots/eda/boxplot_" + save_name + ".png", dpi=400, bbox_inches='tight')
    plt.show()

    return stats


def analyze_vocabulary(df: pd.DataFrame, text_col: str = 'caption', n_top: int = 10, remove_stopwords: bool = True, save_name=None) -> dict:
    """
    Analyze vocabulary and n-grams in a DataFrame with text captions.
    :param df: DataFrame containing text captions
    :param text_col: Name of column containing text
    :param n_top: Number of top items to show
    :param remove_stopwords: Whether to remove English stopwords
    :param save_name: Name for saving plots
    :returns: Dictionary with analysis results and visualization components
    """
    # Preprocessing
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    punct = set(string.punctuation)

    # Process all text
    all_text = ' '.join(df[text_col].str.lower().tolist())
    tokens = word_tokenize(all_text)

    # Filter tokens
    filtered_tokens = [
        token for token in tokens
        if token not in stop_words and token not in punct
    ]

    # Calculate metrics
    word_freq = Counter(filtered_tokens)
    bigrams = Counter(ngrams(filtered_tokens, 2))
    trigrams = Counter(ngrams(filtered_tokens, 3))

    # Generate plots
    plt.figure(figsize=(18, 8))

    # Word frequency plot
    plt.subplot(1, 3, 1)
    top_words = word_freq.most_common(n_top)
    words, counts = zip(*top_words)
    plt.barh(words[::-1], counts[::-1], color=sns.color_palette("plasma", n_colors=len(top_words)))
    plt.title(f'Top {n_top} Words')
    plt.xlabel('Frequency')

    # Bigram frequency plot
    plt.subplot(1, 3, 2)
    top_bigrams = bigrams.most_common(n_top)
    bigram_labels = [' '.join(bg) for bg, _ in top_bigrams]
    counts = [count for _, count in top_bigrams]
    plt.barh(bigram_labels[::-1], counts[::-1], color=sns.color_palette("plasma", n_colors=len(top_bigrams)))
    plt.title(f'Top {n_top} Bigrams')
    plt.xlabel('Frequency')

    # Trigram frequency plot
    plt.subplot(1, 3, 3)
    top_trigrams = trigrams.most_common(n_top)
    trigram_labels = [' '.join(tg) for tg, _ in top_trigrams]
    counts = [count for _, count in top_trigrams]
    plt.barh(trigram_labels[::-1], counts[::-1], color=sns.color_palette("plasma", n_colors=len(top_trigrams)))
    plt.title(f'Top {n_top} Trigrams')
    plt.xlabel('Frequency')
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=400, bbox_inches='tight')
    plt.show()

    return {
        'num_unique_words': len(word_freq),
        'total_words': len(filtered_tokens),
        'word_frequency': word_freq,
        'top_words': top_words,
        'bigram_frequency': bigrams,
        'top_bigrams': top_bigrams,
        'trigram_frequency': trigrams,
        'top_trigrams': top_trigrams
    }
