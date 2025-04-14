import pandas as pd


def clean_up_csv():
    df = pd.read_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00.csv")

    df.loc[df['model'].isnull(), 'model'] = 'basic'
    df.to_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v1.csv", index=False)

    df.loc[df['freeze_encoder'] == False, 'fine_tune_encoder'] = 'partial'
    df.drop('freeze_encoder', axis='columns', inplace=True)
    df.to_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v2.csv", index=False)

    df.loc[df['epoch.max'].isnull(), 'epoch.max'] = df['epoch']
    df.drop(columns=['epoch'], inplace=True)
    df.to_csv("../results/wandb_export_2025-04-13T21_58_18.628-03_00_v3.csv", index=False)


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
    count_experiments()
