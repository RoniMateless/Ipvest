import config
import subprocess

def extractDNSFeatures(pcap_file_name):
 print("Now extracting features...")
 str1 = config.tshark_location + " -r " + pcap_file_name + " -2 -R \"dns.flags==0x100\" -T fields -e dns.qry.name -e ip.id -e frame.time_epoch -e ip.src -e ip.ttl -E separator=,"
 #str1 = tshark_location + " -r " + fileName + " -2 -R \"(ip.dst == 8.8.8.8 or ip.dst == 54.213.224.174) and dns.flags==0x100\" -T fields -e dns.qry.name -e ip.id -e frame.time_epoch -e ip.src -e ip.ttl -E separator=,"
 print("Now running " + str1)
 output = subprocess.check_output(str1)
 outputFileName = pcap_file_name + config.OUTPUT_ENDING
 with open(outputFileName,"w") as outputFile:
	 outputFile.write(output)
 return outputFileName

def extractDNSFeatures(pcap_file_name):
 print("Now extracting features...")
 str1 = config.tshark_location + " -r " + pcap_file_name + " -2 -R \"dns.flags==0x100\" -T fields -e dns.qry.name -e ip.id -e frame.time_epoch -e ip.src -e ip.ttl -E separator=,"
 #str1 = tshark_location + " -r " + fileName + " -2 -R \"(ip.dst == 8.8.8.8 or ip.dst == 54.213.224.174) and dns.flags==0x100\" -T fields -e dns.qry.name -e ip.id -e frame.time_epoch -e ip.src -e ip.ttl -E separator=,"
 print("Now running " + str1)
 output = subprocess.check_output(str1)
 outputFileName = pcap_file_name + config.OUTPUT_ENDING
 with open(outputFileName,"wb") as outputFile:
	 outputFile.write(output)
 return outputFileName

def extractFeatures(pcap_file_name,outputFileName,internal_ip_addresses,listOfTCPFeatures,extraFeatures):
 print("Now extracting features...")
 joined_list = listOfTCPFeatures+extraFeatures
 joined_list.append('dns.resp.name')
 features_text = " ".join(["-e "+feature for feature in joined_list]) 
 if len(internal_ip_addresses) == 1:
    ip_addresses_string = "ip.addr == "+internal_ip_addresses[0]+ " "
 else:
    ip_addresses_string = "ip.addr == "+internal_ip_addresses[0]+ " or ip.addr == "
    ip_addresses_string += " or ip.addr == ".join(internal_ip_addresses[1:])
 ip_addresses_string = config.captured_ips
 str1 = config.tshark_location + " -r " + pcap_file_name + " -2 -R \"not ipv6 and (dns or tcp or udp) and (not icmp) and ("+ip_addresses_string+")\" -T fields " + features_text + " -E separator=, > "+ outputFileName
 print(str1)
 #str1 = " -r " + pcap_file_name + " -2 -R \"tcp\" -T fields " + features_text + " -E separator=, > " + outputFileName
 #print("Now running " + str1)
 #Create a batch file
 with open("1.bat","w") as outputFile:
  outputFile.write(str1)
 #with open(outputFileName,"w") as outputFile:
 subprocess.run("1.bat")
 #output = subprocess.run(str1)
 #os.system(str1)
 #args = ["-r",pcap_file_name,"-2","-R","\"tcp\"","-T","fields"]
 #args+=listOfTCPFeatures
 #args+=["-E","separator=,",">",outputFileName]
 #print(args)
 #os.execv(tshark_location,args)
 #subprocess.call(str1)
 #with open(outputFileName,"wb") as outputFile:
 #	 outputFile.write(output)
 return outputFileName


