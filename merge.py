import glob
import sys
import subprocess
import os
mergecpap_location = "\"c:\\Program Files\\Wireshark\\mergecap.exe\""
input_dir = sys.argv[1]
list_of_files = sorted(glob.glob(input_dir + "\*.pcap"))
to_merge_amount = 10
#create subdir at target
try:
  merged_dir = input_dir+"\\Merged"
  os.mkdir(merged_dir)
except Exception as E:
  print(E)

for i in range(0,len(list_of_files),to_merge_amount):
    sub_list = list_of_files[i:i+to_merge_amount]
    private_first_file_name = sub_list[0].split("\\")[-1]
    output_file_name = merged_dir+"\\"+ private_first_file_name.replace(".pcap","_merged_"+str(i)+".pcap")
    str1 = mergecpap_location + " -w " + output_file_name + " " + " ".join(sub_list)
    print(str1)
    output = subprocess.check_output(str1)
    print(output)





