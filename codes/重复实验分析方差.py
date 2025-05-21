import pandas as pd

df = pd.read_csv("llm_repeat_result/heatmap_values.csv")

max_value = df.select_dtypes(include='number').values.max()

positions = (df == max_value)

for row_index, row in positions.iterrows():
    for col_name, is_max in row.items():
        if is_max:
            print(f"最大值 {max_value} 位于：行标签 = {row_index}, 列标签 = '{col_name}'")

#Male, Animal Husbandry, "A bright and shining star to all femininity but only one girl for \"Dicky.\" Knows every inch of the road east of Ames, at least a mile.",
#Royalty: 0.2 0.6 0.2 0.8 0.8 0.8 0.8 0.2 0.8 0.8

numeric_df = df.select_dtypes(include='number')

column_sums = numeric_df.sum()

max_sum_column = column_sums.idxmax()    
max_sum_value = column_sums.max()        

print(f"求和最大的列是：'{max_sum_column}'，总和为：{max_sum_value}")

#Male, Civil Engineering, "This youth succeeded in passing up Phys. the first time he tried. An advertisement of his pony will be found in the advertising section. Favorite motto: R'I/e = M",
