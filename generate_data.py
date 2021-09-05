
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import time
PCAP_PATH = '.\JSON\packets.json'
#PCAP_FOLDER_PATH = '.\JSON_140219'
PCAP_FOLDER_PATH = '.\JSON_13_18_02'
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def read_jsons_from_folder(path_to_json, max_files = 10000000):
    # this finds our json files
    json_files = [os.path.join(path_to_json, pos_json) for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json') and not pos_json.endswith('statistics.json')]
    print(json_files)
    print('No of JSON files: ', len(json_files))

    dfs = [pd.read_json(js) for js in json_files[:max_files]] # ,orient='index'

    df = pd.concat(dfs, ignore_index=True)

    return df

def feature_engineering(df):

    print(df['A_ip'].groupby(df['A_ip']).size())
#    print(df['related_dns_ipid'].groupby(df['related_dns_ipid']).size())

    '''
    print(df['Recommended_TS_group_server'].groupby(df['Recommended_TS_group_server']).size())
    df['COUNTER'] = 1  # initially, set that counter to 1.
    group_data = df.groupby(['A_ip', 'Recommended_TS_group'])['COUNTER'].sum()  # sum function
    print(group_data)
    
    #print(df.groupby(df[['A_ip','Recommended_TS_group']]).size())
    '''
    print("After Filter data: X: {}".format(df.shape))

    df['start_time'] = pd.to_datetime(df['start_time'],unit='s')
    df = df.sort_values('start_time')  # sort data by start time

    # print(df['related_dns_ipid'].count())
    # get the first and last time in data
    first_time = df['start_time'].iloc[0]
    last_time = df['start_time'].iloc[-1]
    print("First time in data", first_time)
    print("Last time in data", last_time)

    A_IP2Id = {w: i for i, w in enumerate(df["A_ip"].unique())}
    print("No of Classes:", len(A_IP2Id))
    Id2A_IP = {v: k for k, v in A_IP2Id.items()}
    print(Id2A_IP)

    # Set A_ip to be gt
    gt_df = df["A_ip"].replace(A_IP2Id)
    gt_df = gt_df.rename("label")

    #print('Start Feature Engineering...')
    # parse related_dns_ipid from hex to int
    #df['related_dns_ipid'] = df['related_dns_ipid'].apply(lambda x: int(str(x), 16))
    # Split B_IP to classes
    df1 = pd.DataFrame(df['B_ip'].apply(lambda x: x.split('.')[:4]))
    df[['B_ip_Class_A', 'B_ip_Class_B', 'B_ip_Class_C', 'B_ip_Class_D']] = pd.DataFrame(df1['B_ip'].values.tolist(),
                                                                                        index=df1.index)

    # returns the elapsed milliseconds since the start of the program
    def millis(x):
        dt = x - first_time
        ms = int((dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0)

        return max(ms, 1)

    df['time_elapsed'] = df['start_time'].apply(millis)
    df['time_elapsed_in_sec'] = df['time_elapsed'] / 1000

    #df['first_tcp_ts'] = df['first_tcp_ts'].astype('float64')
#    df['first_tcp_ts_server'] = df['first_tcp_ts_server'].astype('float64')
    #df['related_dns_ipid'] = df['related_dns_ipid'].astype('int')
    df['log_first_tcp_ts'] = np.log(df['first_tcp_ts']+0.0001)
    df['log_first_tcp_ts_server'] = np.log(df['first_tcp_ts_server'] + 0.0001)
    df['normalized_tcp_ts'] = np.sqrt(df['time_elapsed']) + df['first_tcp_ts']
    df['normalized_tcp_ts_server'] = np.sqrt(df['time_elapsed']) + df['first_tcp_ts_server']
    df['normalized_related_dns_ipid'] = np.sqrt(df['time_elapsed']) + df['related_dns_ipid']
    df['log_related_dns_ipid'] = np.log(df['related_dns_ipid']+0.0001)



    oh_categorical = ["http_user_agent", "http.cookie", "x509sat.printableString"]
    for f in oh_categorical:
        feat = pd.get_dummies(df[f])
        df = pd.concat([df, feat], axis=1)

    label_categorical = ["B_ip_host"]  # , "http.host"]
    for f in label_categorical:
        df[f] = LabelEncoder().fit_transform(df[f])


    df.drop(['A_ip', 'B_ip', 'A_port', 'end_time', 'http_user_agent','http.cookie', 'x509sat.printableString','A_ip_host', 'stream_state', 'protocol_L4',
             'http.host', 'connection_id', 0], axis=1, inplace=True)#'start_time',

    df = df.astype({"B_ip_Class_A": 'uint8', "B_ip_Class_B": 'uint8', "B_ip_Class_C": 'uint8', "B_ip_Class_D": 'uint8',
                    "A_to_B_bytes": 'uint64', "A_to_B_pkts": 'uint32', "first_tcp_ts": 'uint32', 'ip_ttl': 'uint8',
                    "related_dns_ipid": 'uint16', "normalized_related_dns_ipid": 'float32', "time_duration": 'float32',
                    "time_elapsed": 'uint64', "time_elapsed_in_sec": 'uint64', 'normalized_tcp_ts': 'float64', 'normalized_tcp_ts_server': 'float64',
                    "log_first_tcp_ts": 'float16', "log_related_dns_ipid": 'float16', 'Recommended_TS_group' : 'int8', #'start_time': 'uint16',
                    'Recommended_TS_group_server': 'int8', "first_tcp_ts_server": 'uint32', "log_first_tcp_ts_server": 'float16',
                    'B_ip_host': 'uint32', "B_port": 'uint32', "B_to_A_bytes": 'uint64', "B_to_A_pkts": 'uint32'})
    gt_df = gt_df.astype({"label": 'uint8'})

    #print(df.dtypes)

    return df, gt_df, Id2A_IP

def generate_features(data=None):
    t0 = time.time()
    if data:
        df = pd.DataFrame.from_dict(data, orient='index')
    #else:
    #    df = read_jsons_from_folder(PCAP_FOLDER_PATH, 5)

    #row_data = df.to_dict()
    print("Original data: X: {}".format(df.shape))

    # Filter non-physical devices rows
    #white_list = ['10.100.102.3', '10.100.102.5', '10.100.102.8', '10.100.102.10', '10.100.102.11', '10.100.102.15', '10.100.102.17']
    #win_list = ['10.100.102.11', '10.100.102.17']
    #linux_list = ['10.100.102.3', '10.100.102.5', '10.100.102.8','10.100.102.10', '10.100.102.15']
    #debug_list = ['10.100.102.11', '10.100.102.17','10.100.102.18','10.100.102.19', '10.100.102.10', '10.100.102.14', '10.100.102.15', '10.100.102.16','10.100.102.3', '10.100.102.5', '10.100.102.8']
    #debug2_list = ['10.100.102.15']
    #df = df.loc[(df['start_time'] > '2019-02-16 16:00:32.316690922') & (df['start_time'] < '2019-02-18 22:00:32.316690922')]
    #filter_df = df.loc[(df['A_ip'].isin(white_list)) & (df['protocol_L4'] == 'TCP') & (df['A_to_B_bytes'].astype('int') > 0)]
    #filter_df = df.loc[(df['protocol_L4'] == 'TCP') & (df['A_to_B_bytes'].astype('int') > 0)]
    filter_df = df.replace('', 0)  # , inplace=True)  # replace empty cells with 0 //np.NaN
    #print (filter_df.head())

    filter_df['ip_ttl'] = filter_df['ip_ttl'].astype('uint8')
    #filter_df['first_tcp_ts'] = filter_df['first_tcp_ts'].astype('int64')
    #filter_df['first_tcp_ts'] = filter_df['first_tcp_ts'].replace(0, 1000000000)  # , inplace=True)  # replace empty cells with 0 //np.NaN
    win_df = win_gt = Id2WinIP = linux_df = linux_gt = Id2LinuxIP = other_os_df = other_os_gt = Id2other_os_IP = None
    win_df = filter_df.loc[(filter_df['ip_ttl'] > 120) & (filter_df['ip_ttl'] < 130)]# & (filter_df['related_dns_ipid'] != 0)] #win and consider only rows with related_dns_ipid
    linux_df = filter_df.loc[(filter_df['ip_ttl'] > 10) & (filter_df['ip_ttl'] < 65)]# & (filter_df['first_tcp_ts'] != 0)]#linux and consider only rows with tcp ts
    other_os_df = filter_df.loc[((filter_df['ip_ttl'] > 0) & (filter_df['ip_ttl'] < 10)) | ((filter_df['ip_ttl'] > 65) & (filter_df['ip_ttl'] < 120)) | (filter_df['ip_ttl'] > 130)]

    #print ("all linux size:",linux_df.shape)
    #df_dbg = linux_df.loc[(linux_df['first_tcp_ts'] == 1000000000)]
    #print("no tcp_ts size:", df_dbg.shape)

    if not win_df.empty:
        print ("win feature_engineering...")
        win_df, win_gt, Id2WinIP = feature_engineering(win_df)
    if not linux_df.empty:
        print("linux feature_engineering...")
        linux_df, linux_gt, Id2LinuxIP = feature_engineering(linux_df)
    if not other_os_df.empty:
        print("other os feature_engineering...")
        other_os_df, other_os_gt, Id2other_os_IP = feature_engineering(other_os_df)
    t1 = time.time()
    #print("Time:", ('%.2fs' % (t1 - t0)).lstrip('0'))

    return win_df, win_gt, Id2WinIP, linux_df, linux_gt, Id2LinuxIP, other_os_df, other_os_gt, Id2other_os_IP




#X1,y1, Id2A_IP1, X2, y2, Id2A_IP2 = generate()