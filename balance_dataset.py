import pandas as pd
import random
seed=9999
random.seed(seed)
# 用于获取多标签分类中最少数量的那个标签，和数量
def get_min_count_label(data_list:pd.DataFrame,labels_columns:list):
    data_list_sum=data_list.sum()
    min_count_label=labels_columns[0]
    min_count=data_list_sum[min_count_label]
    for label in labels_columns:
        if data_list_sum[label]<min_count:
            min_count_label=label
            min_count=data_list_sum[label]
    return min_count_label,min_count


def get_balanced_dataset(origin_data:pd.DataFrame,MIN_NUM:int,labels_columns:list,seed=9999)->pd.DataFrame:
    random.seed(seed)
    new_data=origin_data.copy()
    min_label,min_count=get_min_count_label(new_data,labels_columns)
    while min_count<MIN_NUM:
        add_source=origin_data[origin_data[min_label]==1]
        new_data.loc[len(new_data)]=add_source.iloc[random.randint(0,len(add_source)-1)]
        min_label,min_count=get_min_count_label(new_data,labels_columns)
    new_data.sample(frac=1)
    return new_data



for i in range(2):
    data_=pd.read_csv('/disk1/wangjialei/datasets/OIA-ODIR/Training_Set/Annotation/training_annotation_(English)_single.csv')
    data_=get_balanced_dataset(data_,1000,['N','D','G','C','A','H','M','O'],seed=random.randint(0,324))
    data_.to_csv(f'./balanced_labels_1000{i}.csv',index=False)
    print(data_.sum())