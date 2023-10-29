import pandas as pd

mlm = pd.read_csv("MLM.csv")
hlm = pd.read_csv("HLM.csv")

mlm["HLM"] = hlm["HLM"]

mlm.to_csv("result.csv", index=False)