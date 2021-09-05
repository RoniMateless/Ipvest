import dns
import string
import json
import collections
import uuid
import math
import os
import os.path
from collections import OrderedDict
import subprocess
import sys
import random
import glob

################ Files related to the program #############
import config
import general
from features import listOfTCPFeatures,extraFeatures
import output
import tshark
import connections
import algo

write_lists = False

def readFeaturesLine(line,listOfTCPFeatures,extraFeatures):
   #split_line = line.replace("\n","").split("\t")
   split_line = line.rstrip().split(",")
   #print(len(split_line))
   #print(len(extraFeatures))
   #print(len(listOfTCPFeatures)+len(extraFeatures))
   dict1 = {}
   '''
   print(len(split_line)) 
   print(len(listOfTCPFeatures)+len(extraFeatures))
   print(split_line)
   print(listOfTCPFeatures)
   print(extraFeatures)
   '''
   #if len(split_line) > len(listOfTCPFeatures)+len(extraFeatures)+1:
   #   return None,split_line
   #quit()
   '''
   print("ffffffff")
   print(line)
   print(split_line)
   print("line of split_line:",len(split_line))
   print("length:",len(listOfTCPFeatures)+len(extraFeatures))
   quit()
   '''
   for index1,feature in enumerate(listOfTCPFeatures+extraFeatures):

    dict1[feature] = split_line[index1]

   dict1['dns.resp.name'] = split_line[len(listOfTCPFeatures)+len(extraFeatures):]
   #print(dict1)
   return dict1,split_line

def getPacketDirection(dict1,internal_ip_addresses):
   #Identify TCP or UDP
   if dict1['ip.src'] in internal_ip_addresses and dict1['ip.dst'] in internal_ip_addresses:
       return None,None
   elif dict1['ip.src'] in internal_ip_addresses:
       packet_direction = "A_to_B"
       sending_computer = dict1['ip.src_host']
   elif dict1['ip.dst'] in internal_ip_addresses:
       packet_direction = "B_to_A"
       sending_computer = dict1['ip.dst_host']
   else:
       return None,None#packet without internal computer, skip packet
   return packet_direction,sending_computer

def getConnectionID(dict1,packet_direction):
   if dict1['tcp.srcport'] != "": #TCP found!
          if packet_direction == "A_to_B":
             connection_id = "_".join([dict1[x] for x in ['ip.src','tcp.srcport','ip.dst','tcp.dstport']])
          elif packet_direction == "B_to_A":
             connection_id = "_".join([dict1[x] for x in ['ip.dst','tcp.dstport','ip.src','tcp.srcport']])
          else:
               raise Exception("Unkown direction!")
   elif dict1['udp.srcport'] != "": #UDP found!
          if packet_direction == "A_to_B":
             connection_id = "_".join([dict1[x] for x in ['ip.src','udp.srcport','ip.dst','udp.dstport']])
          elif packet_direction == "B_to_A":
             connection_id = "_".join([dict1[x] for x in ['ip.dst','udp.dstport','ip.src','udp.srcport']])
          else:
               raise Exception("Unkown direction!")
   else:
        raise Exception("No TCP nor UDP!")
   return connection_id

def updateConnections(dns_dict,tcp_dict,pcap_file_name,statistics,internal_ip_addresses,TS_dict,TS_max_index_list,TS_dict_server,TS_max_server_index_list):
 print("ttttttttttttttt")
 pcap_file_name = pcap_file_name
 #pcap_file_name = "\""+pcap_file_name+"\""
 print("f")
 lines_skipped = 0
 tcp_file_name = pcap_file_name + config.OUTPUT_ENDING  
 if config.overwrite or not os.path.isfile(tcp_file_name):
    tcp_file_name = tshark.extractFeatures(pcap_file_name,tcp_file_name,internal_ip_addresses,listOfTCPFeatures,extraFeatures)
 with open(tcp_file_name) as inputFile:
  lines = inputFile.readlines()
  for line in lines:
    #find five tuple on dict
   dict1,split_line = readFeaturesLine(line,listOfTCPFeatures,extraFeatures)
   dns.updateDNS(dns_dict,dict1)
   if dict1 == None: #bad line
      #print("Line skipped!")
      lines_skipped+=1
      print(lines_skipped)
      continue

   connection_id = ""
   packet_direction,sending_computer = getPacketDirection(dict1,internal_ip_addresses)
   if packet_direction == None:
      continue

   if dict1['tcp.flags.reset'] == '1':#Reset on a new stream is not accepted
          continue#skip stream

   tcp_dict_entry = OrderedDict()
   connection_id = getConnectionID(dict1,packet_direction)

   if connection_id in tcp_dict: #existing connection
       closed = connections.checkTimeoutAndPossiblyCloseOldStream(tcp_dict,dict1,connection_id)
       if closed:
             connections.openNewStream(dns_dict,tcp_dict,dict1,tcp_dict_entry,connection_id,split_line,packet_direction,listOfTCPFeatures,extraFeatures,write_lists)
   else: #five tuple wasn't in the tcp_dict, it's a new connection!
       connections.openNewStream(dns_dict,tcp_dict,dict1,tcp_dict_entry,connection_id,split_line,packet_direction,listOfTCPFeatures,extraFeatures,write_lists)
       
   sending_computer = dict1['ip.src_host']
   tcp_dict_entry = tcp_dict[connection_id]
   bytes_count = int(dict1['frame.len'])
   #print("bytes_count",bytes_count)
   #update packets from A to B
   if packet_direction == "A_to_B":
     #print("Before",tcp_dict_entry["A_to_B_pkts"])
     tcp_dict_entry["A_to_B_pkts"] = str(int(tcp_dict_entry["A_to_B_pkts"])+1)
     tcp_dict_entry["A_to_B_bytes"] = str(int(tcp_dict_entry["A_to_B_bytes"])+bytes_count)
     if write_lists:
      tcp_dict_entry['packet_direction'].append('A_to_B')
     #print("After",tcp_dict_entry["A_to_B_pkts"])
   elif packet_direction == "B_to_A":
     tcp_dict_entry["B_to_A_pkts"] = str(int(tcp_dict_entry["B_to_A_pkts"])+1)
     tcp_dict_entry["B_to_A_bytes"] = str(int(tcp_dict_entry["B_to_A_bytes"])+bytes_count)
     if write_lists:
      tcp_dict_entry['packet_direction'].append('B_to_A')
   else:
     raise Exception("No packet direction found!")

   #update x509sat.printableString and http.cookie
   if dict1["x509sat.printableString"] != "":
       tcp_dict_entry["x509sat.printableString"] += "|" + dict1["x509sat.printableString"]
   if dict1["http.cookie"] != "":
       tcp_dict_entry["http.cookie"] += "|" + dict1["http.cookie"]

   #update tcp timestamp for client side
   if tcp_dict[connection_id]['first_tcp_ts'] == "":#we didn't got timestamp
    if dict1['tcp.options.timestamp.tsval'] != "":#and got it now
     if packet_direction == "A_to_B":#and it's of the right direction
      tcp_dict_entry['first_tcp_ts'] = dict1['tcp.options.timestamp.tsval']

   #update recommanded tcp timestamp for client side
   if tcp_dict[connection_id]['first_tcp_ts'] != "":#we got timestamp
       timestamp_value = tcp_dict[connection_id]['first_tcp_ts']
       choosen_group = algo.updateTSGroups(TS_dict,timestamp_value,TS_max_index_list)
       tcp_dict_entry['Recommended_TS_group'] = choosen_group

   #update tcp timestamp for server side
   if tcp_dict[connection_id]['first_tcp_ts_server'] == "":#we didn't got timestamp
    if dict1['tcp.options.timestamp.tsval'] != "":#and got it now
     if packet_direction == "B_to_A":#and it's of the right direction
      tcp_dict_entry['first_tcp_ts_server'] = dict1['tcp.options.timestamp.tsval']

   #update recommanded tcp timestamp for server side
   if tcp_dict[connection_id]['first_tcp_ts_server'] != "":#we got timestamp
       timestamp_value = tcp_dict[connection_id]['first_tcp_ts_server']
       choosen_group = algo.updateTSGroups(TS_dict_server,timestamp_value,TS_max_server_index_list)
       tcp_dict_entry['Recommended_TS_group_server'] = choosen_group

   #update ip.ttl
   if tcp_dict[connection_id]['ip_ttl'] != "":#we didn't got ttl
    if dict1['ip.ttl'] != "":#and got it now
     if packet_direction == "A_to_B":#and it's of the right direction
      tcp_dict_entry['ip_ttl'] = dict1['ip.ttl']

   #update duration
   duration = float(dict1['frame.time_epoch'])-float(tcp_dict_entry['start_time'])
   assert(duration>=0.0)
   tcp_dict_entry['time_duration']=str(duration)
       
   #update ip.id
   if write_lists:
    print("---------------------")
    tcp_dict_entry['ip_id'].append(dict1['ip.id'])

   #update tcp.options.timestamp.tsval
   #tcp_dict_entry['tcp_options_timestamp_tsval'].append(dict1['tcp.options.timestamp.tsval'])

   #update frame time and frame length
   if write_lists:
    tcp_dict_entry['frame_time_epoch'].append(dict1['frame.time_epoch'])
    tcp_dict_entry['frame_len'].append(dict1['frame.len'])

   #update user agent
   tcp_dict_entry['http_user_agent'] = dict1['http.user_agent']
 
   #update tcp_dict with the tcp_dict_entry
   tcp_dict[connection_id]=tcp_dict_entry

   #update end_time and possibly close
   if dict1['tcp.flags.fin'] == '1':
     tcp_dict[connection_id]=tcp_dict_entry
     #print("** ",connection_id," **")
     #print(tcp_dict[connection_id])
     connections.closeStream(tcp_dict,dict1,connection_id,'Closed_by_FIN')
   elif dict1['tcp.flags.reset'] == '1':
     tcp_dict_entry['end_time'] = dict1['frame.time_epoch']
     assert(float(tcp_dict_entry['start_time']) <= float(tcp_dict_entry['end_time'])) 
     tcp_dict[connection_id]=tcp_dict_entry
     connections.closeStream(tcp_dict,dict1,connection_id,'Closed_by_RESET')

   #update statistics
   hostA = tcp_dict_entry['A_ip_host'] 
   hostB = tcp_dict_entry['B_ip_host']
   if "hosts_with_names" not in statistics:
    statistics["hosts_with_names"] = OrderedDict()
   if "hosts_without_names" not in statistics:
    statistics["hosts_without_names"] = OrderedDict()

   for word in [hostA,hostB]:
    if general.containtsText(word):
     if word in statistics["hosts_with_names"]:
      statistics["hosts_with_names"][word]+=1
     else:
      statistics["hosts_with_names"][word]=1
    else:
     if word in statistics["hosts_without_names"]:
      statistics["hosts_without_names"][word]+=1
     else:
      statistics["hosts_without_names"][word]=1

   user_agent = tcp_dict_entry['http_user_agent']
   if user_agent != "": 
    if 'http_user_agent' in tcp_dict_entry:
      if 'user_agents' not in statistics:
         statistics['user_agents'] = {}
      if sending_computer not in statistics['user_agents']:
         statistics['user_agents'][sending_computer] = {user_agent:1}
      else:
         if user_agent in statistics['user_agents'][sending_computer]:
            statistics['user_agents'][sending_computer][user_agent] += 1 
         else:
            statistics['user_agents'][sending_computer][user_agent] = 1

#if __name__ == "__main__":
def extract_from_pcap():
 #input_dir,number_of_files = general.parse_command_line(sys.argv)
 #input_dir, number_of_files = r'D:\DeNAT\Roni_AP_140219\Merged', 1000000000000
 input_dir, number_of_files = r'.'+os.sep+'Input'+os.sep, 1000000000000
 input_dir, number_of_files = config.captured_dir+os.sep, 1000000000000
 internal_ip_addresses = general.read_internal_ips()
 #to add os.path.seperator
 list_of_files = sorted(glob.glob(input_dir+"*.pcap"))[:number_of_files]
 print(list_of_files)
 amount_of_files = len(list_of_files)
 list1 = []
 tcp_dict = OrderedDict()
 dns_dict = OrderedDict()
 TS_dict = OrderedDict()
 TS_max_index_list = [-1]
 TS_dict_server = OrderedDict()
 TS_max_server_index_list = [-1]

 statistics = OrderedDict()
 #listOfTCPFeatures,extraFeatures = features.getFeatures()

 for index1,pcap_file_name in enumerate(list_of_files):
  print("Now working on ",index1," out of ",amount_of_files)
  updateConnections(dns_dict,tcp_dict,pcap_file_name,statistics,internal_ip_addresses,TS_dict,TS_max_index_list,TS_dict_server,TS_max_server_index_list)


 #list1 = list(tcp_dict.values())
 output.writeFiles(tcp_dict,statistics)
 return tcp_dict
