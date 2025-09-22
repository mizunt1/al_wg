import pandas as pd
from waterbirds_dataset import count_groups
mdata = pd.read_csv('data/waterbirds_v1.0/metadata.csv')

# 0,1,2,3 is train, 4 is test, 5 is validation

# group 3 train construction 
land = 0
water = 1

mdata['split'] = -1
lw = mdata[(mdata['y']==land) & (mdata['place'] ==water)] 
ww = mdata[(mdata['y']==water) & (mdata['place'] ==water)] 
wl = mdata[(mdata['y']==water) & (mdata['place'] ==land)] 
ll = mdata[(mdata['y']==land) & (mdata['place'] ==land)] 

data_groups = [ww, wl, ll, lw]
def change_split_id(data_list, counts, used, group_id):
    for count, data, start in zip(counts, data_list, used):
        indexes = data[start:count+start].index.values
        mdata.loc[list(indexes),['split']]=group_id

# train data
# G0
used = [0,0,0,0]
counts = [0, 531, 0, 531]
change_split_id(data_groups, counts, used, 0)
used = [i + j for i,j in zip(used, counts)]


# G1 to G3
counts = [500, 0, 500, 0]
change_split_id(data_groups, counts, used, 1)
used  = [i + j for i,j in zip(used, counts)]  
change_split_id(data_groups, counts, used, 2)
used  = [i + j for i,j in zip(used, counts)]
counts = [532, 0, 532, 0]
change_split_id(data_groups, counts, used, 3)
# change split id for group 0
used  = [i + j for i,j in zip(used, counts)]


###### do the same for test ##########
counts = [200, 0, 0, 0]
change_split_id(data_groups, counts, used, 4)
used  = [i + j for i,j in zip(used, counts)]  
counts = [0, 200, 0, 0]
change_split_id(data_groups, counts, used, 5)
used  = [i + j for i,j in zip(used, counts)]
counts = [0, 0, 200, 0]
change_split_id(data_groups, counts, used, 6)
# change split id for group 0
used  = [i + j for i,j in zip(used, counts)]
counts = [0, 0, 0, 200]
change_split_id(data_groups, counts, used, 7)

# change split id for val
used  = [i + j for i,j in zip(used, counts)]
counts = [100, 100, 2488, 1174]
change_split_id(data_groups, counts, used, 8)


for i in range(0,8):
    print('group i', i)
    print(len(mdata[mdata['split'] == i]))

mdata.to_csv('data/waterbirds_v1.0/metadata_v7.csv')
