import sys
import random
import string

def parse_command_line(command_line_list):
 if len(command_line_list) < 2:
  print("")
  print("Usage: python ",sys.argv[0]," input_dir (number_of_files)")
  print("")
  print("If number_of_files isn't present it will run on all files in input_dir") 
  print("which may take a lot of time...")
  quit()
 
 input_dir = sys.argv[1]
 if len(sys.argv) > 2:
  number_of_files = int(sys.argv[2])
 else:
  number_of_files = 100000000000
 return input_dir,number_of_files

def read_internal_ips():
 internal_ip_addresses = []
 with open('ips.csv') as inputFile:
  lines = inputFile.readlines()
  for line in lines:
   internal_ip = line.rstrip().split(",")[0]
   internal_ip_addresses.append(internal_ip)
 return internal_ip_addresses

def containtsText(str1):
 return any(c.isalpha() for c in str1)

# Disable
def disablePrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def randomString():
  return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))


