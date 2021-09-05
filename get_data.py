TS_dict = {}
#timestamp_delta = 1000
timestamp_delta = 100000
def updateTSGroups(TS_dict,timestamp_value,TS_max_index_list):
    TS_max_index = TS_max_index_list[0]
    related_groups = []
    choosen_group = -1
    for group in TS_dict:
        #print("g0",group)
        #print("g1",TS_dict[group])
        #print("g2",timestamp_value)
        if abs(float(TS_dict[group]) - float(timestamp_value)) <= timestamp_delta:
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
        print("More than one TS group detected!")
        print("group number","group's last TS value")
        for group in related_groups:
            print(group,TS_dict[group])
        choosen_group = -1

    TS_max_index_list[0] = TS_max_index
    return choosen_group

with open('FromMyHouseToAMachineWithEightContainersEightServersEach_b.pcap.csv') as inputFile:
    lines = inputFile.readlines()

list1 = []
TS_dict = {}
results = {}
TS_max_index_list = [0]
for line in lines:
    time,port,ts = line.split("\t")
    ts = int(ts)
    port_index = int(port)-2000
    if 0<=port_index<=100:
        print(line)
        #updateTSGroups(TS_dict,timestamp_value,TS_max_index_list):
        recommended = updateTSGroups(TS_dict,ts,TS_max_index_list)
        if recommended not in results:
            results[recommended] = set()
        results[recommended].add(port_index)
        
print(results)
