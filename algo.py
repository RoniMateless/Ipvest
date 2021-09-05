import config
def updateTSGroups(TS_dict,timestamp_value,TS_max_index_list):
    TS_max_index = TS_max_index_list[0]
    related_groups = []
    choosen_group = -1
    for group in TS_dict:
        #print("g0",group)
        #print("g1",TS_dict[group])
        #print("g2",timestamp_value)
        if abs(float(TS_dict[group]) - float(timestamp_value)) <= config.timestamp_delta:
           related_groups.append(group)
    if len(related_groups) == 0:
       #create a new group
       TS_max_index+=1
       TS_dict[TS_max_index] = timestamp_value
       choosen_group = TS_max_index
    elif len(related_groups) == 1:
        #update froup
        choosen_group = related_groups[0]
        TS_dict[choosen_group] = timestamp_value
    elif len(related_groups) > 1:
        #don't decide
        #print("More than one TS group detected!")
        #print("group number","group's last TS value")
        #for group in related_groups:
        #    print(group,TS_dict[group])
        choosen_group = -1

    TS_max_index_list[0] = TS_max_index
    return choosen_group

