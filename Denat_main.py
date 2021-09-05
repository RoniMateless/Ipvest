from extractor import extract_from_pcap
from generate_data import generate_features
from DeNAT_HClustering import make_clustering, set_number_of_groups
from output import index_to_ES
import config
import json
import time
import csv
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import number_of_clusters
from functools import reduce
from PIL import Image
from io import BytesIO

def plot_scatter_dates(x, y, ylabel, o_file, start_id=0):
    # fig = plt.figure(figsize=(7,4))
    fig = plt.figure(figsize=(7, 4))

    for id, (i, j) in enumerate(zip(x, y)):
        axs = plt.plot_date(i, j, label='id %s' % str(id+start_id), ms=2)

    plt.xlabel("Time", fontsize=10)
    plt.yscale('log')
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(rotation=45)#[i for i in range(14)],
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # tic a  `ks along the bottom edge are off
        left=False)  # ticks along the top edge are off
    plt.legend(loc="lower left")
    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_minor_formatter(myFmt)

    plt.show()
    fig.savefig(o_file, dpi=600)  # , bbox_inches='tight')
    # save figure
    # (1) save the image in memory in PNG format
    png1 = BytesIO()
    #fig = axs.get_figure()
    fig.savefig(png1, format='png', dpi=600)#, bbox_inches='tight')

    # (2) load this image into PIL
    png2 = Image.open(png1)

    hr_file = o_file + '.tiff'
    # (3) save as TIFF
    png2.save(hr_file, resolution=600)
    png1.close()
    #plt.savefig(o_file, bbox_inches='tight', dpi=100)

def plot_clustering(data, s, yfield, ylabel, o_file):
    d = defaultdict(list)
    for x in s.items():
        d[x[1]].append(x[0])

    values = []
    times = []
    for vd, con_idx_l in d.items():
        row1 = []
        row2 = []
        for i in con_idx_l:
            if data[i][yfield]:
                row1.append(int(data[i][yfield]))
                row2.append(data[i]['start_time'])
        values.append(row1)
        times.append(row2)

    #list_of_datetimes = [[datetime.fromtimestamp(float(i)) for i in sub_l] for sub_l in times]
    #print(list_of_datetimes)

    dates = [mdates.date2num(sub_l) for sub_l in times]
    plot_scatter_dates(x= dates, y=values, ylabel=ylabel, o_file=o_file, start_id=100)


def plot_gt_values(list_of_tuples, ylabel, o_file):
    gt_d = defaultdict(list)
    for i in list_of_tuples:
        #k = i[0].rsplit('.', 1)[0]
        #if k > '2019-02-18 13:55' and k < '2019-02-18 14:07' and i[1]>0:
        #if k > '2019-02-14 00:00' and k < '2019-02-14 07:00' and i[1] > 0:
        gt_d[i[2]].append((i[0], i[1]))
    values = []
    times = []

    for ip, sub_l in gt_d.items():
        values.append([int(i[1]) for i in sub_l])
        times.append([i[0] for i in sub_l])

    dates = [mdates.date2num(sub_l) for sub_l in times]
    plot_scatter_dates(x= dates, y=values, ylabel=ylabel, o_file=o_file)


def clean_flows(data):
    filtered_data = data.copy()
    print (len(filtered_data))
    for id,con in data.items():
        filtered_data[id]['start_time'] = datetime.fromtimestamp(float(con['start_time']))
        if con['related_dns_ipid']:
            filtered_data[id]['related_dns_ipid'] = int(con['related_dns_ipid'], 16)
        if con['first_tcp_ts']:
            filtered_data[id]['first_tcp_ts'] = int(con['first_tcp_ts'])
        if con['first_tcp_ts_server']:
            filtered_data[id]['first_tcp_ts_server'] = int(con['first_tcp_ts_server'])
        if con['protocol_L4'] == 'UDP' or (con['ip_ttl'] == '128' and not con['related_dns_ipid']) or\
                (con['ip_ttl'] == '64' and not con['first_tcp_ts'])or int(con['A_to_B_bytes']) < 1 or \
                (con['ip_ttl'] == '64' and not con['first_tcp_ts_server']) or con['ip_ttl'] == '255' or \
                str(filtered_data[id]['start_time']) > '2019-02-14 06:00:00' or\
                str(filtered_data[id]['start_time']) < '2019-02-14 00:00:00':
                #str(filtered_data[id]['start_time']) > '2019-02-15 16:07:00' or\
                #str(filtered_data[id]['start_time']) < '2019-02-15 15:58:00':
            del filtered_data[id]
    print(len(filtered_data))

    return filtered_data


def preprocessing_server(data):
    for id, con in data.items():
        digits = str(con['first_tcp_ts_server'])[:3]
        if digits.startswith('15'):
            con['A_ip'] = digits
        else:
            con['A_ip'] = digits[:2]

    return data

#FLOWS_PATH = '.\JSON_EP_SHORT\packets.json'
#FLOWS_PATH = '.\JSON\packets.json'
#FLOWS_PATH = '.\JSON_Servers\packets.json'
FLOWS_PATH = '.\JSON_EP_FULL\packets.json'
if __name__ == "__main__":
    t0 = time.time()
    data=None
    #data = extract_from_pcap()
    with open(FLOWS_PATH) as f:
        data = json.load(f)

    data = clean_flows(data)
    #data = preprocessing_server(data)

    #for id, con in data.items():
    #    print(datetime.fromtimestamp(float(con['start_time'])))

    X_win, y_win, Id2A_IP_win, X_linux, y_linux, Id2A_IP_linux, X_other_os, y_other_os, Id2A_IP_other_os = generate_features(data)

    # Write results to a file
    f = open('denat_results___.csv', 'a')
    measures_fields = ['OS', 'Features', 'Algorithm', 'n_clusters', 'Purify', 'ARI', 'Precision', 'Recall', 'F_1', 'F_0.5', 'F_0.2', 'RI', 'homogeneity', 'completeness', 'v-measure-1 ', 'v-measure-0.5', 'v-measure-0.2', 'time']#, 'Silhouette', 'db_score','bic','Time']
    print(measures_fields)
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(measures_fields)

    #for win_clusters, linux_clusters in ((4,6), (5,7), (6,8), (7,9), (8,10)):
    #for win_clusters, linux_clusters in ((4, 8),):
        # Win
    if not X_win.empty:

        list_of_tuples = [tuple((con['start_time'], con['related_dns_ipid'], con['A_ip']))
                          for con in data.values() if con['A_ip'] in Id2A_IP_win.values() and con['related_dns_ipid'] and con['ip_ttl'] == '128']

        #plot_gt_values(list_of_tuples, ylabel='related-dns-ip-id value', o_file='win-home-gt')
        #n_clusters, times, values = number_of_clusters.bp_alg(list_of_tuples, value_thresh = 1000)#pow(2,16)/10)
        #print ("Number of clusters", n_clusters)
        #dates = [mdates.date2num(sub_l) for sub_l in times]
        #plot_scatter_dates(x=dates, y=values, ylabel='Number of entities', o_file='bp-win')

        #X_win = X_win[['related_dns_ipid']]# Baseline
        X_win = X_win[['time_elapsed', 'related_dns_ipid', 'B_ip_host']]# XGBoost, 'B_ip_Class_A', 'B_port', 'B_ip_Class_D', 'B_ip_Class_C', '|Washington', '|Google Trust Services']]#, 'time_duration']]#, 'A_to_B_bytes']]'time_elapsed'
        #X_win.drop(['start_time'], axis=1, inplace=True)
        #set_number_of_groups(X=X_win, gt=y_win)
        #s = make_clustering(X=X_win, gt=y_win, start_device_index=0, n_clusters=3,Id2IP=Id2A_IP_win, OS_name = 'Win', features_name = 'Supervised', file_writer=writer) #Related IP ID

        #plot_clustering(data, s, yfield='related_dns_ipid', ylabel='related-dns-ip-id values', o_file='win-home-clustering')

        #for x in s.items():
        #    data[x[0]]['virtual_device_id'] = str(x[1])


    # Linux
    if not X_linux.empty:
        '''
        list_of_tuples = [tuple((con['start_time'], con['first_tcp_ts_server'], con['A_ip']))#first_tcp_ts_server
                          for con in data.values() if con['A_ip'] in Id2A_IP_linux.values() and con['first_tcp_ts_server'] and con['ip_ttl'] == '64']

        #plot_gt_values(list_of_tuples, ylabel='tcp_ts values', o_file='lin-endpoints-gt')
        plot_gt_values(list_of_tuples, ylabel='tcp_ts server values', o_file='lin-servers-gt')#tcp_ts server values
        '''
        list_of_tuples = [tuple((con['start_time'], con['first_tcp_ts'], con['A_ip']))  # first_tcp_ts_server
                          for con in data.values() if
                          con['A_ip'] in Id2A_IP_linux.values() and con['first_tcp_ts'] and con[
                              'ip_ttl'] == '64']

        plot_gt_values(list_of_tuples, ylabel='tcp-ts values', o_file='lin-home-gt')  # tcp_ts server values

        #n_clusters, times, values = number_of_clusters.bp_alg(list_of_tuples, value_thresh = 100000)#pow(2,32)/pow(2,20))
        #print ("Number of clusters", n_clusters)
        #dates = [mdates.date2num(sub_l) for sub_l in times]
        #plot_scatter_dates(x=dates, y=values, ylabel='Number of entities', o_file='bp-lin')
        #n_clusters = number_of_clusters_static_alg(list_of_tuples, pow(2,20))
        #n_clusters = number_of_clusters_dynamic_alg(list_of_tuples, -pow(2,20), pow(2,20))

        #X_linux = X_linux[['first_tcp_ts']]  # Baseline
        #X_linux = X_linux[['time_elapsed', 'first_tcp_ts', 'B_ip_host']]
        #X_linux = X_linux[['normalized_tcp_ts', 'Recommended_TS_group', 'B_ip_host']]  # , 'B_ip_Class_A']]#, 'time_elapsed']]#, 'B_ip_host']]
        #X_linux = X_linux[['normalized_tcp_ts_server', 'first_tcp_ts_server', 'B_ip_host']]  # , 'B_ip_Class_A']]#, 'time_elapsed']]#, 'B_ip_host']] #'normalized_tcp_ts_server', 'first_tcp_ts_server', 'B_ip_host'
        X_linux = X_linux[['normalized_tcp_ts', 'first_tcp_ts',  'B_ip_host']]#, 'B_ip_Class_A']]#, 'time_elapsed']]#, 'B_ip_host']]
        #set_number_of_groups(X=X_linux, gt=y_linux)
        #s = make_clustering(X=X_linux, gt=y_linux, start_device_index=100, n_clusters=config.linux_n_clusters, Id2IP=Id2A_IP_linux)
        s = make_clustering(X=X_linux, gt=y_linux, start_device_index=100, n_clusters=8, Id2IP=Id2A_IP_linux, OS_name = 'Linux', features_name = 'Supervised', file_writer=writer) # First tcp ts
        #plot_clustering(data, s, yfield='first_tcp_ts_server', ylabel='tcp_ts server values', o_file='lin-servers-clustering')
        plot_clustering(data, s, yfield='first_tcp_ts', ylabel='tcp-ts values',o_file='lin-home-clustering')

        #for x in s.items():
        #    data[x[0]]['virtual_device_id'] = str(x[1])

    # Other OS
    #if not X_other_os.empty:
    #    s = make_clustering(X=X_other_os, gt=y_other_os, start_device_index=200, n_clusters=config.other_os_n_clusters, Id2IP=Id2A_IP_other_os)
    #    for x in s.items():
    #        data[x[0]]['virtual_device_id'] = str(x[1])


    # remove excluded_ips from data
    #saved_list = [con for con in list(data.values()) if con['B_ip_host'] not in config.excluded_ips]
    #with open("test1.json", "w") as f:
    #    f.write(json.dumps(saved_list, indent=4))

    # Write list(data.values()) to Elastic
    #index_to_ES(saved_list)
    #t1 = time.time()
    #print("Overall Time:", ('%.2fs' % (t1 - t0)).lstrip('0'))

