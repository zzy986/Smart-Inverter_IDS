import random

import pandas as pd

import numpy as np

df = pd.read_csv('window_inverter_11_14.csv')


df['Timestamp'] = pd.to_datetime(df['Timestamp'])


specific_date = '2023-11-14'
start_hour_random = 10
end_hour_random = 11


column_to_modify = 'DC_Voltage'
column_to_modify_1='DC_Current'


mask_random = (df['Timestamp'].dt.date == pd.to_datetime(specific_date).date()) & \
       (df['Timestamp'].dt.hour >= start_hour_random) & \
       (df['Timestamp'].dt.hour < end_hour_random)
random.seed(42)


for idx in df[mask_random].index:
    random_factor = np.random.choice(np.arange(0.1, 2.0, 0.01))  # 生成 [0.1, 0.7] 范围内的随机数
    print(random_factor)
    df.loc[idx, column_to_modify] *= random_factor
    df.loc[idx, column_to_modify_1] *= random_factor




start_hour_over = 8
end_hour_over = 9
mask_over = (df['Timestamp'].dt.date == pd.to_datetime(specific_date).date()) & \
       (df['Timestamp'].dt.hour >= start_hour_over) & \
       (df['Timestamp'].dt.hour < end_hour_over)
df.loc[mask_over, column_to_modify] = 1.5*df.loc[mask_over, column_to_modify]
df.loc[mask_over, column_to_modify_1] = 1.5*df.loc[mask_over, column_to_modify_1]


start_hour_org = 14
end_hour_org = 15
start_hour_delay = 15
end_hour_delay = 16

# 创建时间段的掩码
mask_delay = (df['Timestamp'].dt.date == pd.to_datetime(specific_date).date()) & \
              (df['Timestamp'].dt.hour >= start_hour_delay) & \
              (df['Timestamp'].dt.hour < end_hour_delay)

mask_org = (df['Timestamp'].dt.date == pd.to_datetime(specific_date).date()) & \
           (df['Timestamp'].dt.hour >= start_hour_org) & \
           (df['Timestamp'].dt.hour < end_hour_org)

# 复制数据
df.loc[mask_delay, column_to_modify] = df.loc[mask_org, column_to_modify].values
df.loc[mask_delay, column_to_modify_1] = df.loc[mask_org, column_to_modify_1].values




start_hour_replay = 13
end_hour_replay = 14


df_re = pd.read_csv('window_inverter_10_13.csv')
mask_replay = (df['Timestamp'].dt.date == pd.to_datetime(specific_date).date()) & \
       (df['Timestamp'].dt.hour >= start_hour_replay) & \
       (df['Timestamp'].dt.hour < end_hour_replay)


df_re['Timestamp'] = pd.to_datetime(df_re['Timestamp'])

re_date='2023-10-13'
mask_oct = (df_re['Timestamp'].dt.date == pd.to_datetime(re_date).date()) & \
       (df_re['Timestamp'].dt.hour >= start_hour_replay) & \
       (df_re['Timestamp'].dt.hour < end_hour_replay)



df.loc[mask_replay, column_to_modify] = df_re.loc[mask_oct, column_to_modify].values
df.loc[mask_replay, column_to_modify_1] = df_re.loc[mask_oct, column_to_modify_1].values



df.to_csv('attack_14.csv', index=False)

"""
mask_source = (df['Timestamp'].dt.date == pd.to_datetime(specific_date).date()) & \
             (df['Timestamp'].dt.hour >= 12) & \
             (df['Timestamp'].dt.hour < 13)

mask_target = (df['Timestamp'].dt.date == pd.to_datetime(specific_date).date()) & \
             (df['Timestamp'].dt.hour >= 13) & \
             (df['Timestamp'].dt.hour < 14)

# 定义需要替换的列（排除时间戳以保持原时间）
columns_to_replace = ['DC_Voltage', 'DC_Current']

# 确保源数据和目标数据行数一致
df[mask_source].shape[0] == df[mask_target].shape[0]
    # 使用NumPy数组直接替换以避免索引对齐问题
df.loc[mask_target, columns_to_replace] = df.loc[mask_source, columns_to_replace].values


print(df[300:400])
# 保存修改后的数据
df.to_csv('attack_14.csv', index=False)

"""







