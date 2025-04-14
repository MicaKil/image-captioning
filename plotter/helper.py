import pandas as pd

df = pd.read_csv("../report/final/wandb_export_2025-04-13T21_58_18.628-03_00.csv")

df.loc[df['model'].isnull(), 'model'] = 'basic'
df.to_csv("../report/final/wandb_export_2025-04-13T21_58_18.628-03_00_v1.csv", index=False)

df.loc[df['freeze_encoder'] == False, 'fine_tune_encoder'] = 'partial'
df.drop('freeze_encoder', axis='columns', inplace=True)
df.to_csv("../report/final/wandb_export_2025-04-13T21_58_18.628-03_00_v2.csv", index=False)

df.loc[df['epoch.max'].isnull(), 'epoch.max'] = df['epoch']
df.drop(columns=['epoch'], inplace=True)
df.to_csv("../report/final/wandb_export_2025-04-13T21_58_18.628-03_00_v3.csv", index=False)
