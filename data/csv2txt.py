import pandas as pd

# name_list = ["HLM_train", "HLM_val", "MLM_train", "MLM_val"]
# for name in name_list:
#     df = pd.read_csv(f"{name}.csv")
#     with open(f"{name}.txt", "w") as f:
#         for _, rows in df.iterrows():
#             f.write(f"{rows['SMILES']} {rows[name[:3]]}\n")
    

test_df = pd.read_csv("test.csv")
with open(f"data_pred.txt", "w") as f:
    for _, rows in test_df.iterrows():
        f.write(f"{rows['SMILES']}\n")