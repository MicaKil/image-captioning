import pandas as pd


def clean_up_results_csv():
    df = pd.read_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00.csv")

    df.loc[df['model'].isnull(), 'model'] = 'basic'
    df.to_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v1.csv", index=False)

    df.loc[df['freeze_encoder'] == False, 'fine_tune_encoder'] = 'partial'
    df.drop('freeze_encoder', axis='columns', inplace=True)
    df.to_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v2.csv", index=False)

    df.loc[df['epoch.max'].isnull(), 'epoch.max'] = df['epoch']
    df.drop(columns=['epoch'], inplace=True)
    df.to_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v3.csv", index=False)


def clean_up_val_loss_csv():
    df = pd.read_csv("../plots/results/csv_source/attn_8_wandb_export_2025-04-16T00_09_29.345-03_00.csv")
    columns_to_drop = df.filter(regex=".*_step.*").columns
    new = df.drop(columns=columns_to_drop)
    new.to_csv("../plots/results/attn_8_wandb_export_2025-04-16T00_09_29.345-03_00_v2.csv", index=False)
    columns_to_keep = new.filter(regex="val_loss$").columns
    new2 = new[columns_to_keep]
    new2.to_csv("../plots/results/attn_8_wandb_export_2025-04-16T00_09_29.345-03_00_v3.csv", index=False)
    columns_to_drop = [col for col in new2.columns if new2[col].notnull().sum() == 1]
    new3 = new2.drop(columns=columns_to_drop)
    new3.to_csv("../plots/results/attn_8_wandb_export_2025-04-16T00_09_29.345-03_00_v4.csv", index=False)

    df = pd.read_csv("../plots/results/csv_source/lstm_38_wandb_export_2025-04-16T00_16_18.415-03_00.csv")
    columns_to_drop = df.filter(regex=".*_step.*").columns
    new = df.drop(columns=columns_to_drop)
    new.to_csv("../plots/results/lstm_38_wandb_export_2025-04-16T00_16_18.415-03_00_v2.csv", index=False)
    columns_to_keep = new.filter(regex="val_loss$").columns
    new2 = new[columns_to_keep]
    new2.to_csv("../plots/results/lstm_38_wandb_export_2025-04-16T00_16_18.415-03_00_v3.csv", index=False)
    columns_to_drop = [col for col in new2.columns if new2[col].notnull().sum() == 1]
    new3 = new2.drop(columns=columns_to_drop)
    new3.to_csv("../plots/results/lstm_38_wandb_export_2025-04-16T00_16_18.415-03_00_v4.csv", index=False)

    df = pd.read_csv("../plots/results/csv_source/attn_50_wandb_export_2025-04-16T00_09_29.345-03_00.csv")
    columns_to_drop = df.filter(regex=".*_step.*").columns
    new = df.drop(columns=columns_to_drop)
    new.to_csv("../plots/results/attn_50_wandb_export_2025-04-16T00_09_29.345-03_00_v2.csv", index=False)
    columns_to_keep = new.filter(regex="val_loss$").columns
    new2 = new[columns_to_keep]
    new2.to_csv("../plots/results/attn_50_wandb_export_2025-04-16T00_09_29.345-03_00_v3.csv", index=False)
    columns_to_drop = [col for col in new2.columns if new2[col].notnull().sum() == 1]
    new3 = new2.drop(columns=columns_to_drop)
    new3.to_csv("../plots/results/attn_50_wandb_export_2025-04-16T00_09_29.345-03_00_v4.csv", index=False)


def merge_csv_by_columns(file_paths, output_path):
    # Read all CSV files into a list of DataFrames
    dataframes = [pd.read_csv(file_path) for file_path in file_paths]

    # Concatenate the DataFrames by adding columns
    merged_df = pd.concat(dataframes, axis=1)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to {output_path}")


def remove_duplicate_columns(file_path, output_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Remove duplicate columns
    df = df.T.drop_duplicates().T

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"Duplicate columns have been removed and saved to {output_path}")


def count_experiments():
    df = pd.read_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v3.csv")
    print("Total number of experiments:", len(df))

    # Count experiments for encoder = 'resnet50' and decoder = 'LSTM'
    resnet50_lstm_count = len(df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'LSTM')])

    # Count experiments for encoder = 'resnet50' and decoder = 'Attention'
    resnet50_attention_count = len(df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'Attention')])

    # Count experiments for encoder = 'swin' and decoder = 'LSTM'
    swin_lstm_count = len(df[(df['encoder'] == 'swin') & (df['decoder'] == 'LSTM')])

    # Count experiments for encoder = 'swin' and decoder = 'Attention'
    swin_attention_count = len(df[(df['encoder'] == 'swin') & (df['decoder'] == 'Attention')])

    # Print the results
    print(f"resnet50 + LSTM: {resnet50_lstm_count}")
    print(f"resnet50 + Attention: {resnet50_attention_count}")
    print(f"swin + LSTM: {swin_lstm_count}")
    print(f"swin + Attention: {swin_attention_count}")

    # Sanity check
    assert resnet50_lstm_count + resnet50_attention_count + swin_lstm_count + swin_attention_count == len(
        df), "Counts do not match total number of experiments"

    flickr8k_count = len(df[df['dataset.name'] == 'flickr8k'])
    coco_count = len(df[df['dataset.name'] == 'coco'])

    print(f"flickr8k: {flickr8k_count}")
    print(f"coco: {coco_count}")

    assert flickr8k_count + coco_count == len(df), "Counts do not match total number of experiments"

    print(f"resnet50 + LSTM + Flickr8k: {len(df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'LSTM') & (df['dataset.name'] == 'flickr8k')])}")
    print(
        f"resnet50 + Attention + Flickr8k: {len(df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'Attention') & (df['dataset.name'] == 'flickr8k')])}")
    print(f"swin + Attention + Flickr8k: {len(df[(df['encoder'] == 'swin') & (df['decoder'] == 'Attention') & (df['dataset.name'] == 'flickr8k')])}")
    print(f"resnet50 + LSTM + Coco: {len(df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'LSTM') & (df['dataset.name'] == 'coco')])}")
    print(f"resnet50 + Attention + Coco: {len(df[(df['encoder'] == 'resnet50') & (df['decoder'] == 'Attention') & (df['dataset.name'] == 'coco')])}")
    print(f"swin + Attention + Coco: {len(df[(df['encoder'] == 'swin') & (df['decoder'] == 'Attention') & (df['dataset.name'] == 'coco')])}")


if __name__ == "__main__":
    # file_paths = [
    #     "../plots/results/attn_8_wandb_export_2025-04-16T00_09_29.345-03_00_v4.csv",
    #     "../plots/results/attn_50_wandb_export_2025-04-16T00_09_29.345-03_00_v6.csv",
    #     "../plots/results/lstm_38_wandb_export_2025-04-16T00_16_18.415-03_00_v4.csv"
    # ]
    # output_path = "../plots/results/val_loss.csv"
    # merge_csv_by_columns(file_paths, output_path)

    remove_duplicate_columns("../plots/results/csv_source/val_loss.csv", "../plots/results/csv_source/val_loss_v2.csv")
