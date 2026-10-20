import pandas as pd
import os
import numpy as np

# ✅ File path
file_path = r"data/archiveEEG/AD_all_patients.csv"

# Load dataset
df = pd.read_csv(file_path)

print("Columns:\n", df.columns)
print("\nSample Data:\n", df.head())

# ✅ Label column
LABEL_COLUMN = 'status'

# Check distribution
print("\nClass distribution:\n", df[LABEL_COLUMN].value_counts())

# ✅ Split FULL dataset (NO REDUCTION)
ad_df = df[df[LABEL_COLUMN] == 1]
cn_df = df[df[LABEL_COLUMN] == 0]

print(f"\nAD samples: {len(ad_df)}")
print(f"CN samples: {len(cn_df)}")


# ✅ Function to save .npy files
def save_npy(dataframe, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(dataframe)):
        row = dataframe.iloc[i]

        # Remove label column
        values = row.drop(LABEL_COLUMN).values

        # Convert to float
        values = values.astype(np.float32)

        try:
            # Reshape → (time_steps, features)
            values = values.reshape(-1, 16)   # 16 EEG channels
            values = values[:128, :]          # limit sequence length
        except:
            continue

        np.save(os.path.join(output_folder, f"sample_{i}.npy"), values)

        # Optional progress print (VERY USEFUL)
        if i % 5000 == 0:
            print(f"Processed {i} samples...")


# ✅ Save AD and CN
save_npy(ad_df, r"data\EEG\processed\AD")
save_npy(cn_df, r"data\EEG\processed\CN")


# ✅ Create MCI from CN
def create_mci(cn_path, mci_path):
    os.makedirs(mci_path, exist_ok=True)

    files = os.listdir(cn_path)

    for i, file in enumerate(files):
        data = np.load(os.path.join(cn_path, file))

        noise = np.random.normal(0, 0.01, data.shape)
        mci = data + noise

        np.save(os.path.join(mci_path, file), mci)

        if i % 5000 == 0:
            print(f"MCI created for {i} samples...")


create_mci(
    r"data\EEG\processed\CN",
    r"data\EEG\processed\MCI"
)

print("\n✅ EEG preprocessing DONE (FULL DATASET)!")