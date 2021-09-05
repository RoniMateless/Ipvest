import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from generate_data import generate_features
from matplotlib import pyplot
import pandas as pd
from extractor import extract_from_pcap
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from utility import plot_confusion_matrix
from datetime import datetime
from collections import defaultdict
from PIL import Image
from io import BytesIO
from sklearn.metrics import f1_score, accuracy_score

classifiers = [
    # MLPClassifier(hidden_layer_sizes=(int(MAX_FEATURES/2),),batch_size=32),
    ('XGBoost', XGBClassifier())
#    ('RandomForest', RandomForestClassifier(n_estimators=100))
]
#X = pd.read_pickle('roni_home_data.pkl')
#y = pd.read_pickle('roni_home_labels.pkl')

#X = X[['first_tcp_ts', 'A_to_B_bytes', 'B_ip_host', 'time_elapsed', 'B_ip_Class_C', 'B_ip_Class_B']]#, 'related_dns_ipid', 'A_to_B_bytes', 'B_ip_host', 'time_duration', 'B_ip_Class_C', 'B_ip_Class_B', 'B_port']]
#X.drop(['A_to_B_pkts', 'B_to_A_pkts', 'B_ip_Class_D'], axis=1, inplace=True)

def make_classification(X, y, Id2A_IP):
    X.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)
    '''
    train_len = int(0.8*len(X))
    X_train, X_test = X.iloc[:train_len], X.iloc[train_len:]
    y_train, y_test = y.iloc[:train_len], y.iloc[train_len:]
    '''
    f1_scores = []
    accuracy_scores = []
    train_test = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in train_test.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #print("Training on: X_train: {}, X_test: {}, y_train: {}, y_test: {}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
        #print(y_test.groupby(y_test).size())

        #print (X_test)
        #for name, model in classifiers:
        model = XGBClassifier()
        important_features = np.array([])

        # fit model on training data
        #model.fit(X, y)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, predicted))
        f1_scores.append(f1_score(y_test, predicted, average='weighted'))

    row_measures = [np.mean(accuracy_scores), np.std(accuracy_scores), np.mean(f1_scores), np.std(f1_scores)]

    print(row_measures)
    # feature importance
    #print(model.feature_importances_)
    #print (model.get_booster().get_score(importance_type='weight'))
    #print(model.get_booster().get_score(importance_type='total_gain'))
    #print(model.get_booster().get_score(importance_type='total_cover'))

    # plot feature importance
    X_sliced = [elem[:28] for elem in X.columns.get_values()]
    #important_features = pd.Series(data=model.feature_importances_,index=X_sliced)
    if not np.isnan(model.feature_importances_).any():
        important_features = zip(X_sliced, model.feature_importances_.tolist())

    #important_features.sort_values(ascending=False,inplace=True)

    return important_features

def plot_features(list_of_lists, file):
    d = defaultdict(list)
    for sublist in list_of_lists:
        for i, j in sublist:
            d[i].append(j)
    #print(d)
    index_l = []
    avg_l = []
    for k, v in d.items():
        index_l.append(k)
        avg_l.append(np.average(v))

    s = pd.Series(avg_l, index=index_l)
    return s
    #nlargest = s.nlargest(n=20, keep='first')
    #inner_plot_features(nlargest, file)

def inner_plot_features(df, o_file):

    #print (features)
    #print (features.values)
    #ax = features.plot(kind='barh')  # , title='%s features importance (%s)' % (os_name, name))

    ax = df.plot()  # , title='%s features importance (%s)' % (os_name, name))

    pyplot.xticks(range(len(df.index)), df.index, rotation=45)
    ax.set_xlabel("Time range")
    ax.set_ylabel("Feature importance value")

    pyplot.show()
    fig = ax.get_figure()
    #fig.savefig(o_file, bbox_inches='tight', dpi=600)
    # save figure
    # (1) save the image in memory in PNG format
    png1 = BytesIO()
    fig.savefig(png1, format='png')

    # (2) load this image into PIL
    png2 = Image.open(png1)

    # (3) save as TIFF
    png2.save(o_file, resolution=600)
    png1.close()


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
                (con['ip_ttl'] == '64' and not con['first_tcp_ts_server']) or con['ip_ttl'] == '255':# or \
                #str(filtered_data[id]['start_time']) > '2019-02-14 01:30:00' or\
                #str(filtered_data[id]['start_time']) < '2019-02-14 00:00:00':
            del filtered_data[id]
    print(len(filtered_data))

    return filtered_data


import json
from datetime import timedelta
from functools import reduce

def classification_by_time_period(data, deltatime):
    times = []
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.sort_values('start_time')  # sort data by start time
    df.set_index(df['start_time'] ,inplace=True)
    first_time = datetime(2009, 2, 13, 21, 27,0)
    date_time_str = 'Feb 13 2019 9:27PM'
    first_time = datetime.strptime(date_time_str, '%b %d %Y %I:%M%p')
    date_time_str = 'Feb 18 2019 6:38PM'
    last_time = datetime.strptime(date_time_str, '%b %d %Y %I:%M%p')

    win_df_l = []
    lin_df_l = []

    while first_time < last_time:
        second_time = first_time + deltatime
        con_list = df.loc[first_time:second_time]['connection_id'].tolist()
        data_range = {con:data[con] for con in con_list}
        if data_range:
            X_win, y_win, Id2A_IP_win, X_linux, y_linux, Id2A_IP_linux, X_other_os, y_other_os, Id2A_IP_other_os = generate_features(data_range)

            X_win.drop(['start_time'], axis=1, inplace=True)
            X_linux.drop(['start_time'], axis=1, inplace=True)

            if not X_win.empty:
                X_win = X_win[['related_dns_ipid']]
                features = make_classification(X_win, y_win, Id2A_IP_win)
                if features:
                    win_df_l.append(features)
            if not X_linux.empty:
                X_linux = X_linux[['first_tcp_ts']]
                features = make_classification(X_linux, y_linux, Id2A_IP_linux)
                if features:
                    lin_df_l.append(features)

        first_time = second_time

    #win_s = plot_features(win_df_l, "win-mean-34")
    #lin_s = plot_features(lin_df_l, "lin-mean-67")

    return win_s, lin_s



def onetime_classification(data):
    X_win, y_win, Id2A_IP_win, X_linux, y_linux, Id2A_IP_linux, X_other_os, y_other_os, Id2A_IP_other_os = generate_features(data)
    X_win.drop(['start_time'], axis=1, inplace=True)
    X_linux.drop(['start_time'], axis=1, inplace=True)

    #features = make_classification(X_win, y_win, Id2A_IP_win, "Windows")
    #plot_features(features, "f_importance_win")
    features = make_classification(X_linux, y_linux, Id2A_IP_linux, "Linux")
    plot_features(features, "f_importance_linux")


data=None
#FLOWS_PATH = '.\JSON_EP_SHORT\packets.json'
FLOWS_PATH = '.\JSON_EP_FULL\packets.json'
#FLOWS_PATH = '.\JSON_6\packets.json'
#FLOWS_PATH = '.\JSON_326\packets.json'
#data = extract_from_pcap()


with open(FLOWS_PATH) as f:
    data = json.load(f)
data = clean_flows(data)
#onetime_classification(data)
#[5,10]#
minutes = [60*24*6]#[10,30,60,60*6, 60*12, 60*24, 60*24*2, 60*24*3, 60*24*4, 60*24*5, 60*24*6]
columns = ['10 minutes']#['10 minutes', '30 minutes', '1 hour', '6 hours', '12 hours', '1 day', '2 days', '3 days','4 days', '5 days', '6 days']
win_features = ['time_elapsed', 'related_dns_ipid', 'normalized_related_dns_ipid' ,'B_ip_host', 'B_port','time_duration', 'A_to_B_bytes']
lin_features = ['normalized_tcp_ts', 'first_tcp_ts', 'B_ip_host', 'time_duration', 'time_elapsed','Recommended_TS_group', 'A_to_B_bytes']
w_list = []
l_list = []
for i in minutes:
    win_s, lin_s = classification_by_time_period(data, timedelta(minutes=i))
    series_win = win_s[win_features]
    series_lin = lin_s[lin_features]

    w_list.append(series_win)
    l_list.append(series_lin)

w_df = pd.concat(w_list, axis=1)
l_df = pd.concat(l_list, axis=1)

w_transposed_df = w_df.T
w_transposed_df.index = columns
l_transposed_df = l_df.T
l_transposed_df.index = columns

inner_plot_features(w_transposed_df, "w-fi-trend.tiff")
inner_plot_features(l_transposed_df, "l-fi-trend.tiff")

