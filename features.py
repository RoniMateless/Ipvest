def getFeatures():
 listOfTCPFeatures = ["frame.number","ip.src","tcp.srcport","ip.dst","tcp.dstport","frame.time_epoch","ip.ttl","frame.time_epoch","tcp.flags.fin","tcp.flags.reset","frame.len","ip.id","tcp.options.timestamp.tsval","ip.src_host","ip.dst_host","http.user_agent","udp.srcport","udp.dstport","dns.flags","dns.qry.name","http.cookie","x509sat.printableString"]

 extraFeatures = []
 with open("fields.csv") as fields_text:
  lines = fields_text.readlines()
  for line in lines:
   line = line.rstrip().split(',')
   toTake,field_name = line[0],line[3]
   if toTake != "":
     extraFeatures.append(field_name)

 print("extraFeatures selected: ",extraFeatures)
 return listOfTCPFeatures,extraFeatures


listOfTCPFeatures,extraFeatures = getFeatures()
