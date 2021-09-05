import config
import general

def closeStream(tcp_dict,dict1,old_stream_name,reason_to_close):
          tcp_dict_entry = tcp_dict[old_stream_name]
          tcp_dict_entry['end_time'] = dict1['frame.time_epoch']
          assert(float(tcp_dict_entry['start_time']) <= float(tcp_dict_entry['end_time'])) 
          tcp_dict_entry['stream_state'] = reason_to_close
          new_name = old_stream_name + "_" + general.randomString()
          tcp_dict[old_stream_name]['connection_id'] = new_name
          tcp_dict[new_name] = tcp_dict[old_stream_name].copy()
          del tcp_dict[old_stream_name]

def openNewStream(dns_dict,tcp_dict,dict1,tcp_dict_entry,connection_id,split_line,packet_direction,listOfTCPFeatures,extraFeatures,write_lists):

       start_time = dict1['frame.time_epoch']
       tcp_dict_entry['connection_id']=connection_id
       tcp_dict_entry['start_time']=start_time
       tcp_dict_entry['end_time']='-1'
       tcp_dict_entry['A_to_B_pkts'] = '0'
       tcp_dict_entry['B_to_A_pkts'] = '0'
       tcp_dict_entry['A_to_B_bytes'] = '0'
       tcp_dict_entry['B_to_A_bytes'] = '0'
       tcp_dict_entry["x509sat.printableString"] = dict1["x509sat.printableString"]
       tcp_dict_entry["http.cookie"] = dict1["http.cookie"]
       tcp_dict_entry['Recommended_TS_group'] = '-1'#undefined yet...
       tcp_dict_entry['Recommended_TS_group_server'] = '-1'#undefined yet...

       if packet_direction == "A_to_B":
          tcp_dict_entry['A_ip_host'] = dict1['ip.src_host']
          tcp_dict_entry['B_ip_host'] = dict1['ip.dst_host']
          dns_related_domain = dict1['ip.dst_host']
       elif packet_direction == "B_to_A":
          tcp_dict_entry['A_ip_host'] = dict1['ip.dst_host']
          tcp_dict_entry['B_ip_host'] = dict1['ip.src_host']
          dns_related_domain = dict1['ip.src_host']
       else:
          raise Exception("Unknown direction!")

       #original_domain = 
       #print(domain_to_search)
        
       #print("dns_dict")
       #for key in dns_dict['domains_ipids']:
       #  print(dns_dict['domains_ipids'][key])
       #print(dict1)
       #if (config.minimum_ttl_value_for_dns_ipid <= int(dict1['ip.ttl']) <= config.maximum_ttl_value_for_dns_ipid) and dns_related_domain in dns_dict['nicknames']:
       #print("dns_related_domain",dns_related_domain)
       #print(dns_dict['nicknames'])
       for key in dns_dict['nicknames']:
           if dns_related_domain in key:
               dns_related_domain = key
       if dns_related_domain in dns_dict['nicknames']:
          #print("dns_related_domain:",dns_related_domain)
          tcp_dict_entry['related_dns_ipid'] = dns_dict['nicknames'][dns_related_domain]
          #tcp_dict_entry['related_dns_ipid'] = int(dns_dict['nicknames'][dns_related_domain], 16)
       else:
          tcp_dict_entry['related_dns_ipid'] = ""

       if dict1['tcp.srcport'] != "": #TCP found!
          tcp_dict_entry['protocol_L4']='TCP'
          if packet_direction == "A_to_B":
             tcp_dict_entry['A_ip'] = dict1['ip.src']
             tcp_dict_entry['B_ip'] = dict1['ip.dst']
             tcp_dict_entry['A_port'] = dict1['tcp.srcport']
             tcp_dict_entry['B_port'] = dict1['tcp.dstport']
             tcp_dict_entry['first_tcp_ts'] = dict1['tcp.options.timestamp.tsval']
             tcp_dict_entry['first_tcp_ts_server'] = ""
             tcp_dict_entry['ip_ttl'] = dict1['ip.ttl']
          else:
             tcp_dict_entry['A_ip'] = dict1['ip.dst']
             tcp_dict_entry['B_ip'] = dict1['ip.src']
             tcp_dict_entry['A_port'] = dict1['tcp.dstport']
             tcp_dict_entry['B_port'] = dict1['tcp.srcport']
             tcp_dict_entry['first_tcp_ts'] = ""
             tcp_dict_entry['first_tcp_ts_server'] = dict1['tcp.options.timestamp.tsval']
             tcp_dict_entry['ip_ttl'] = ""

       elif dict1['udp.srcport'] != "": #UDP found!
          tcp_dict_entry['protocol_L4']='UDP'
          if packet_direction == "A_to_B":
             tcp_dict_entry['A_ip'] = dict1['ip.src']
             tcp_dict_entry['B_ip'] = dict1['ip.dst']
             tcp_dict_entry['A_port'] = dict1['udp.srcport']
             tcp_dict_entry['B_port'] = dict1['udp.dstport']
             tcp_dict_entry['first_tcp_ts'] = ""
             tcp_dict_entry['first_tcp_ts_server'] = ""
             tcp_dict_entry['ip_ttl'] = dict1['ip.ttl']
          else:
             tcp_dict_entry['A_ip'] = dict1['ip.dst']
             tcp_dict_entry['B_ip'] = dict1['ip.src']
             tcp_dict_entry['A_port'] = dict1['udp.dstport']
             tcp_dict_entry['B_port'] = dict1['udp.srcport']
             tcp_dict_entry['first_tcp_ts'] = ""
             tcp_dict_entry['first_tcp_ts_server'] = ""
             tcp_dict_entry['ip_ttl'] = ""

       if write_lists:
        tcp_dict_entry['ip_id'] = []
        tcp_dict_entry['tcp_options_timestamp_tsval'] = []
        tcp_dict_entry['packet_direction'] = []
        tcp_dict_entry['frame_time_epoch'] = []
        tcp_dict_entry['frame_len'] = []
       tcp_dict_entry['stream_state'] = 'Open'
       #get split_line relevant part
       split_line = split_line[len(listOfTCPFeatures):]#get just the extra features
       for index1,feature in enumerate(extraFeatures):
         tcp_dict_entry[feature] = split_line[index1]

       #enter to tcp_dict
       tcp_dict[connection_id] = tcp_dict_entry

def checkTimeoutAndPossiblyCloseOldStream(tcp_dict,dict1,old_stream_name):
    #last_packet_time = tcp_dict[old_stream_name]['frame_time_epoch'][-1]
    last_packet_time = float(tcp_dict[old_stream_name]['start_time'])+float(tcp_dict[old_stream_name]['time_duration'])
    current_time = dict1['frame.time_epoch']
    if float(current_time) - float(last_packet_time) > config.timeout_time:
          closeStream(tcp_dict,dict1,old_stream_name,'Closed_by_TIMEOUT')
          return True
    return False


