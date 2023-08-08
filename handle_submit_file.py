import pandas as pd
data=pd.read_csv('./test.csv')
index_columns=data.columns
names=[]
for i in data.iterrows():
    name=i[1]['ID'].split('_')[0]
    if name not in names:
        names.append(name)
result=pd.DataFrame(columns=index_columns)
f = lambda x: x if (isinstance(x, float) and x > 0.5) else 0

for id in names:
    temp_df=data[data['ID'].str.contains(id,case=False)]
    min=temp_df['N'].min()
    max=temp_df[['D', 'G', 'C', 'A', 'H', 'M', 'O']].max()
    max['N']=min
    new_row=pd.Series(max,index=result.columns)
    # new_row=new_row.apply(lambda x:f(x) if isinstance(x,(int,float)) else x)
    new_row['ID']=id
    result=result.append(new_row,ignore_index=True)
result['ID']=result['ID'].round().astype(int)


result.to_csv('submit.csv',index=False)