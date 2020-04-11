import pandas as pd


"""
Source:
https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
"""

def random_under_sampling(df):
    # Class count
    count_class_0, count_class_1 = df.Y.value_counts()

    # Divide by class
    df_class_0 = df[df['Y'] == 0]
    df_class_1 = df[df['Y'] == 1]
    
    df_class_0_under = df_class_0.sample(count_class_1)
    df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
    
    print('Random under-sampling:')
    print(df_test_under.Y.value_counts())
    
    return df_test_under
    
def random_over_sampling(df):
    # Class count
    count_class_0, count_class_1 = df.Y.value_counts()

    # Divide by class
    df_class_0 = df[df['Y'] == 0]
    df_class_1 = df[df['Y'] == 1]
    
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
    
    print('Random over-sampling:')
    print(df_test_over.Y.value_counts())
    
    return df_test_over