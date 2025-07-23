import os
from collections import Counter

import pandas as pd

from utils.GeneralTool import GeneralTool

if __name__ == '__main__':
    df_list = []
    base_path = f"{GeneralTool.root_path}/data/su_you_nei_meng_gu_windfarm/time_series"
    for data_path in os.listdir(base_path):
        current_data_path = f"{base_path}/{data_path}"
        df = pd.read_csv(current_data_path)
        df_list.append(df)
    total_df = pd.concat(df_list, axis=0)
    raw_y_distribution = dict(Counter(total_df["target"]).most_common())
    print(f"y distribution: {raw_y_distribution}")