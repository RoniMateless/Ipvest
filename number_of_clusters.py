from datetime import datetime
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error

def plot_grid(x, y, ylabel, marker, o_file):

    # fig = plt.figure(figsize=(7,4))
    plt.figure(figsize=(7, 4))
    print (x ,y)
    col = np.where((np.array(x) == '14:03') | (np.array(x) == '14:06'), 'red', 'blue')

    # c = ['r' for t in x if t == '14:03']np.where(x < '14:03', 'r', 'b')
    plt.scatter(x=x ,y=y ,s=60, marker=marker, color=col  )  # ,edgecolor='k'

    # plt.grid(True)
    plt.xlabel("Time" ,fontsize=10)
    plt.ylabel(ylabel ,fontsize=10)
    plt.xticks(rotation=45)
    # plt.yticks([i for i in range(36)],fontsize=12)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False)  # ticks along the top edge are off
    plt.show()
    # fig = "fig-st-win-bp.png"

    # plt.savefig(o_file, bbox_inches='tight', dpi=100)


def number_of_clusters_static_alg(list_of_tuples, normalized_factor):
    d = defaultdict(list)
    for k, v, v2 in list_of_tuples:
        if v != 0: # value exists
            k_sec = k.rsplit('.', 1)[0]
            k_min = k.rsplit(':', 1)[0]
            k_h = k.rsplit(':', 2)[0]
            d[k_min].append((int(v/normalized_factor), v2))
    print(d)

    max_v = 0
    for k,v in d.items():
        val = [i[0] for i in v]
        max_v = max(max_v, len(set(val)))
    for k,v in d.items():
        val = [i[0] for i in v]
        ip = [i[1] for i in v]
        if len(set(ip)) >= 3:
            print(k, set(val), set(ip))
        #if len(set(val)) >= max_v-1:

    boxes = []
    time = []
    for k, v in d.items():
        #if k > '2019-02-13 21:30' and k < '2019-02-13 22:10':
        #if k > '2019-02-14 09:40' and k < '2019-02-14 10:10':2019-02-14 15:08
        if k > '2019-02-18 13:55' and k < '2019-02-18 14:07':
            val = [i[0] for i in v]
            print (val)
            for i in set(val):
                if i != 16:
                    boxes.append(i)
                    time.append(k.rsplit(' ', 1)[1])
#plot_grid(x=boxes, y=time)
    plot_grid(x=time, y=boxes, ylabel="Box ids", marker='s', o_file= "fig-st-win-bp.png")
    print (max_v)


def bp_alg(list_of_tuples, value_thresh):
    def floor_dt(dt, interval):
        replace = (dt.second // interval) * interval
        return dt.replace(second=replace,microsecond=0)

    value_thresh = value_thresh# minutes
    print ("value_thresh:", value_thresh)
    min_distance = value_thresh*0.5
    d = {}
    gt_d = defaultdict(list)
    # 1. init hash table with t, box_id, min_v, max_v
    for t, v, ip in list_of_tuples:
        if v != 0: # value exists
            t_range = t.replace(second=0, microsecond=0) #t_minutes
            #t_range = t.replace(minute=0, second=0, microsecond=0)  # t_hours
            #t_range = t.replace(microsecond=0)  # t_seconds
            #t_range = floor_dt(t, 60)
            gt_d[t_range].append(ip)
            box_id = int(v / value_thresh)
            if (t_range, box_id) in d:
                min_v, max_v = d[(t_range, box_id)]
                min_v = min(v, min_v)
                max_v = max(v, max_v)
                d[(t_range, box_id)] = (min_v, max_v)
            else:
                d[(t_range, box_id)] = (v, v)

    print("BP", d)

    dis_gt_d = {k:len(set(v)) for k, v in gt_d.items()}
    print("GT", dis_gt_d)

    #2. delete adjacent box ids of the same device
    time2countBoxids = defaultdict(int)
    for k, v in d.items():
        t_range = k[0]
        box_id = k[1]
        #Check if the min_v of box_id minux the max_v of box_id-1 < min_distance AND max_v - min_v < min_distance
        if (t_range, box_id-1) in d:
            min_v, max_v = d[(t_range, box_id)]
            lower_min_v, lower_max_v = d[(t_range, box_id-1)]
            if min_v - lower_max_v < min_distance:# and max_v - min_v < min_distance:
                print ("Found adjacent box ids", t_range, box_id, min_v - lower_max_v, max_v - min_v)
            else:
                print ("Found adjacent box ids, but not close", t_range, box_id, min_v - lower_max_v, max_v - min_v)
                time2countBoxids[t_range] += 1
        else:
            time2countBoxids[t_range] += 1

    print(time2countBoxids)

    print("Length BP:", len(time2countBoxids.keys()))
    print("Length GT:", len(dis_gt_d.keys()))

    # 3. counting
    n_clusters = 0
    times = []
    new_times = []
    values = []
    new_values = []
    for t in sorted(time2countBoxids.keys()):
        n_clusters = max(n_clusters, time2countBoxids[t])
        times.append(t)
        values.append(time2countBoxids[t])

    for t in sorted(time2countBoxids.keys()):
        if time2countBoxids[t] == n_clusters:
            print (t, n_clusters)
    new_times.append(times)
    new_values.append(values)
    # For debug insert gt
    times = []
    values = []
    for t in sorted(dis_gt_d.keys()):
        times.append(t)
        values.append(dis_gt_d[t])
    new_times.append(times)
    new_values.append(values)

    print(new_times)
    print(new_values)
    #for t,i,j in zip(new_times[0], new_values[0], new_values[1]):
    #    if i != j:
    #        print ("Mismatch in time: ", t, i, j)
    print("MSE: ", mean_squared_error(new_values[0], new_values[1]))
    return n_clusters, new_times, new_values



def plot_dyn(x,y):
    plt.figure(figsize=(7, 4))

    for id, (i, j) in enumerate(zip(x, y)):
        plt.plot(i, j, label='id %s' % id)

#    plt.plot_date(x, y)
    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_minor_formatter(myFmt)
    #for i,j in zip(x,y):
    #    plt.scatter(x=x, y=y, s=30, edgecolor='k', marker='s')
        #plt.plot(i, j)#, label='id %s' % i)

    #plt.plot(x[0], y[0], 'r--', x[1], y[1], 'b--', x[2], y[2], 'g--')
    # plt.title("The elbow method",fontsize=16)

    # plt.grid(True)
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("ip-id value", fontsize=10)
    plt.xticks(rotation=45)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False)  # ticks along the top edge are off
    #plt.gcf().autofmt_xdate()
    plt.show()
    fig = "fig-st-win-lines.png"
    #plt.savefig(fig, bbox_inches='tight', dpi=100)


def number_of_clusters_dynamic_alg(list_of_tuples, min_threshold, max_threshold):

    def insert_value_2_lists(list_of_lists, value, id):
        if value != 0:
            for sublist in list_of_lists:
                if value - sublist[-1][0] > min_threshold and value - sublist[-1][0] < max_threshold:
                    sublist.append((value, id))
                    return
            list_of_lists.append([(value, id)])

    print (list_of_tuples)
    list_of_devices = []
    for id, (k, v, v2) in enumerate(list_of_tuples):
        if k > '2019-02-18 13:55:59' and k < '2019-02-18 14:07:00':
            insert_value_2_lists(list_of_devices, v, k)
    print ("Number of devices:", len(list_of_devices))
    lines = []
    time = []
    for sub_l in list_of_devices:
        print (len(sub_l))
        print (sub_l)
        if len(sub_l) > 5:
            for i in sub_l:
                t_time = i[1].rsplit(' ', 1)[1]
                t_time_till_sec = t_time.rsplit('.', 1)[0]
                t_min = t_time.rsplit(':', 1)[0]
                lines.append([i[0] for i in sub_l])
                time.append([i[1].rsplit('.', 1)[0] for i in sub_l])
    out_l = [0 for i in range(len(list_of_tuples))]

    for i, sub_l in enumerate(list_of_devices):
        print (sub_l)
        max_elem = 0
        min_elem = 1000000000000
        for elem in sub_l:
            max_elem = max(max_elem, elem[0])
            min_elem = min(min_elem, elem[0])
        print ("max_min_element", max_elem, min_elem)

    print (time)
    print (lines)

    list_of_datetimes = [[datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in sub_l] for sub_l in time]
    dates = [mdates.date2num(sub_l) for sub_l in list_of_datetimes]
    plot_dyn(x=dates, y=lines)#, ylabel="ip-id value", marker='_', o_file="fig-st-win-lines.png")
    return len(list_of_devices)

