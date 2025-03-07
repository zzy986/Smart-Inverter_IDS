import pandas as pd

# 读取CSV文件
df = pd.read_csv('Sensor_2023_daytime_1.csv')

# 假设日期列名为 'date_column'，并且日期格式为 'YYYY-MM-DD'
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 定义要筛选的特定日期
specific_date = '2023-11-14'  # 指定日期

# 筛选出该日期的数据
filtered_df = df[df['Timestamp'].dt.date == pd.to_datetime(specific_date).date()]

# 打印或保存结果
print(filtered_df)
# 保存为新的CSV文件
filtered_df.to_csv('window_sensor_11_14.csv', index=False)



