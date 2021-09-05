timeout_time = 5*60
OUTPUT_ENDING = ".packets.csv"
tshark_location = "\"c:\\Program Files\\Wireshark\\tshark.exe\""
maxNumberOfDevices = 1000 #unless changed when entering a command
minimum_ttl_value_for_dns_ipid = 80
maximum_ttl_value_for_dns_ipid = 200
timestamp_delta = 100000
overwrite = True
#captured_dir = r'D:\DeNAT\Capture_4_2_19-20190402T213202Z-001\Capture_4_2_19\Merged'
#captured_dir = r'D:\DeNAT\Recording_10_4_19_inside_NAT-20190411T062530Z-001\Recording_10_4_19_inside_NAT\Merged'
#captured_dir = r'D:\DeNAT\test'
captured_dir = r'D:\DeNAT\Roni_AP_140219\Merged'
#captured_dir = r'D:\DeNAT\test'
#captured_dir = r'D:\DeNAT\Roni_140219_Short_endpoints'
#captured_dir = r'D:\DeNAT\test1'
#captured_dir = r'D:\DeNAT\Servers'


#included_ips = ['192.168.1.102', '111.111.111.111'] # delete the second after fix the bug with tshark
# The captured_ips is the relevant field for tshark
captured_ips = "dns or (not (ip.addr == 10.100.102.1 or ip.addr == 239.255.255.250 or ip.addr == 10.100.102.255 or ip.addr == 10.100.102.150 or ip.addr == 230.0.0.1 or ip.addr == 224.0.0.252 or ip.addr == 224.0.0.251) and (ip.addr == 10.100.102.11 or ip.addr == 10.100.102.17 or ip.addr == 10.100.102.18 or ip.addr == 10.100.102.19 or ip.addr == 10.100.102.10 or ip.addr == 10.100.102.15 or ip.addr == 10.100.102.16 or ip.addr == 10.100.102.3 or ip.addr == 10.100.102.5 or ip.addr == 10.100.102.8))"
# The included_ips is the relevant field for packet direction
included_ips = ['10.100.102.2', '10.100.102.3','10.100.102.4','10.100.102.5','10.100.102.6','10.100.102.7','10.100.102.8','10.100.102.9','10.100.102.10','10.100.102.11','10.100.102.12','10.100.102.13','10.100.102.15','10.100.102.16','10.100.102.17','10.100.102.18','10.100.102.19','10.100.102.2','10.100.102.20','10.100.102.21','10.100.102.22','10.100.102.23','10.100.102.24','10.100.102.25']
#included_ips = ['192.168.1.101', '192.168.1.102',  '192.168.1.104', '192.168.1.105', '192.168.1.106', '192.168.1.107', '192.168.1.109', '192.168.1.110'] # haim 10.4.19 101-Liran Android, 102 Roni Android, 104 Win, 105 Intel generic, 106 Haim mobile, 107 Win, 109 Apple, 110 Win Haim Lenovo
excluded_ips = []#['192.168.1.1', '10.100.102.1'] # remove from the display. we need to extract them for dns packets.
win_n_clusters = 4#4
linux_n_clusters = 6#7
other_os_n_clusters = 1
ES_HOST = {"host" : "localhost", "port" : 9200}
#ES_HOST = {"host" : "54.213.224.174", "port" : 80}
ES_INDEX_NAME = 'denat'
ES_TYPE = 'connection'

