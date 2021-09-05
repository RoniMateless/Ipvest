# -2 -R \"dns.flags==0x100\" -T fields -e dns.qry.name -e ip.id -e frame.time_epoch -e ip.src -e ip.ttl -E separator=,"
import config
from collections import OrderedDict
def updateDNS(dns_dict,dict1):
   if len(dns_dict) < 2:
    dns_dict['domains_queries'] = OrderedDict()
    dns_dict['nicknames'] = OrderedDict()
    print("dns_dict created!")
   if dict1:
        query_domain = dict1['dns.qry.name']
        #check if dns query
        if dict1['dns.flags'] == '0x00000100':
           #verifty dns query is from accapted ttl 
           if config.minimum_ttl_value_for_dns_ipid <= int(dict1['ip.ttl']) <=config.maximum_ttl_value_for_dns_ipid:
             ipid = dict1['ip.id']
             dns_dict['domains_queries'][query_domain] = ipid
             #update all relevant nicknames
             dns_dict['nicknames'][query_domain] = ipid
           
        else:#not a query
         response = dict1['dns.resp.name']

         #update nicknames
         #find relevant IPID
         if response != '':
          #find relevant_ipid
          if query_domain in dns_dict['domains_queries']:
            relevant_ipid = dns_dict['domains_queries'][query_domain]
            #and update nicknames
            list_to_update=[query_domain]+response
            for domain in list_to_update:
                dns_dict['nicknames'][domain] = relevant_ipid

