import pandas as pd
from waterbirds_dataset import count_groups
mdata = pd.read_csv('/network/scratch/m/mizu.nishikawa-toomey/waterbird_larger/metadata.csv')


# group 3 train construction 
land = 0
water = 1

mdata_train = mdata[mdata['split'] == 0]
mdata_test = mdata[mdata['split'] == 1]
mdata_val = mdata[mdata['split'] == 2]
lw = mdata_train[(mdata_train['y']==land) & (mdata_train['place'] ==water)] 
ww = mdata_train[(mdata_train['y']==water) & (mdata_train['place'] ==water)] 
wl = mdata_train[(mdata_train['y']==water) & (mdata_train['place'] ==land)] 
ll = mdata_train[(mdata_train['y']==land) & (mdata_train['place'] ==land)] 
mdata['split'] = -1
data_groups = [ww, wl, ll, lw]
def change_split_id(data_list, counts, used, group_id):
    for count, data, start in zip(counts, data_list, used):
        indexes = data[start:count+start].index.values
        mdata.loc[list(indexes),['split']]=group_id

# train data
# G0 wl
mdata.loc[list(wl.index), ['split']] = 0

# G1 lw
mdata.loc[list(lw.index), ['split']] = 1

# G2 ww
mdata.loc[list(ww.index), ['split']] = 2

# G3 ll
mdata.loc[list(ll.index), ['split']] = 3


###### do the same for val ##########
#G4 wl val

lw = mdata_val[(mdata_val['y']==land) & (mdata_val['place'] ==water)] 
ww = mdata_val[(mdata_val['y']==water) & (mdata_val['place'] ==water)] 
wl = mdata_val[(mdata_val['y']==water) & (mdata_val['place'] ==land)] 
ll = mdata_val[(mdata_val['y']==land) & (mdata_val['place'] ==land)] 

# G0 wl
mdata.loc[list(wl.index), ['split']] = 4

# G1 lw
mdata.loc[list(lw.index), ['split']] = 5

# G2 ww
mdata.loc[list(ww.index), ['split']] = 6

# G3 ll
mdata.loc[list(ll.index), ['split']] = 7



###### do the same for test #######
lw = mdata_test[(mdata_test['y']==land) & (mdata_test['place'] ==water)] 
ww = mdata_test[(mdata_test['y']==water) & (mdata_test['place'] ==water)] 
wl = mdata_test[(mdata_test['y']==water) & (mdata_test['place'] ==land)] 
ll = mdata_test[(mdata_test['y']==land) & (mdata_test['place'] ==land)] 

# G0 wl
mdata.loc[list(wl.index), ['split']] = 8

# G1 lw
mdata.loc[list(lw.index), ['split']] = 9

# G2 ww
mdata.loc[list(ww.index), ['split']] = 10

# G3 ll
mdata.loc[list(ll.index), ['split']] = 11


mdata.to_csv('/network/scratch/m/mizu.nishikawa-toomey/waterbird_larger/metadata_largerv2.csv')
