import pandas as pd
import statsmodels.formula.api as smf

# 1. 读取数据
df = pd.read_csv('/Users/liuyaxuan/Desktop/25Spring/25Spring/RA_YilingZhao/t-test.csv')

# 2. 定义维度列表
emotion_cols = [c for c in df.columns if c.lower() != 'gender']

# 3. 循环跑 OLS
for col in emotion_cols:
    # C(Gender) 会生成一个哑变量：Female vs. Male
    formula = f"{col} ~ C(gender)"
    model   = smf.ols(formula, data=df).fit()
    
    print(f"=== Dimension: {col} ===")
    print(model.summary().tables[1])  # 只看系数估计那张表
    print("\n")
