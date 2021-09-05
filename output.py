import json
from elasticsearch import Elasticsearch
import config
import datetime

def writeJsonList(dict,filepath):
 #def chunker(seq, size):
 # return (seq[pos:pos + size] for pos in range(0, len(seq), size))
 #for index1, group in enumerate(chunker(dict, 1000)):
 #fileName = filepath.replace(".json","_"+str(index1)+".json")
 fileName = filepath.replace(".json", ".json")
 with open(fileName,"w") as f:
  f.write(json.dumps(dict,indent=4))

def writeFiles(dict,statistics):
 #create directory if doesn't exist
 import os
 if not os.path.exists("./JSON"):
    os.makedirs("./JSON")
 writeJsonList(dict,"./JSON/packets.json")

 with open("./statistics.json", 'w') as f:
    for chunk in json.JSONEncoder().iterencode(statistics):
        f.write(chunk)

def index_to_ES(list):
    # create ES client, create index
    es = Elasticsearch(hosts=[config.ES_HOST])
    if es.indices.exists(config.ES_INDEX_NAME):
        print("deleting '%s' index..." % (config.ES_INDEX_NAME))
        res = es.indices.delete(index=config.ES_INDEX_NAME)
        print(" response: '%s'" % (res))

    print("creating '%s' index..." % (config.ES_INDEX_NAME))
    request_body = {
        "mappings": {
            "connection": {
                "properties": {
                    "start_time": {
                        "type": "date",
                        "format": "epoch_second"
                    },
                    "related_dns_ipid": {
                        "type": "integer"
                    },
                    "first_tcp_ts": {
                        "type": "long"
                    }
                }
            }
        }
    }
    res = es.indices.create(index=config.ES_INDEX_NAME, body = request_body)
    print(" response: '%s'" % (res))

    # bulk index the data
    print("bulk indexing...")
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    op_dict = {"index": {"_index": config.ES_INDEX_NAME,"_type": config.ES_TYPE}}
    for chunk in chunker(list, 1000):
        bulk_data = []
        for row in chunk:
            bulk_data.append(op_dict)
            bulk_data.append(row)
        res = es.bulk(index=config.ES_INDEX_NAME, body=bulk_data, refresh=True)
        #print(" response: '%s'" % (res))


