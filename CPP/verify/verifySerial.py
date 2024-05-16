import re
import subprocess
import sys
import hashlib

def hash_file(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():

    if len(sys.argv) != 2:
        print("[!] Output file not specified in the argument")
        return -1

    with open(sys.argv[1], 'r') as file:
        content = file.read()

    
    parameters = re.search(r'generalParameter\{fieldLength=(\d+\.\d+),totalStep=(\d+),birdNum=(\d+),randomSeed=(\d+)\}', content)
    field_length = float(parameters.group(1))
    total_step = int(parameters.group(2))
    bird_num = int(parameters.group(3))
    random_Seed = int(parameters.group(4))

    #compute the verification output
    verifyOutputName = sys.argv[1] + ".verify"
    command = "./activeMatter_rawPtr_verifySerial.out {} {} {} {}".format(total_step, bird_num, random_Seed, verifyOutputName)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("[!] Can not compute the verification output with: " + command)
        return -1
    
    #verify with md5
    if hash_file(sys.argv[1]) == hash_file(verifyOutputName):
        print("[*] File: " + sys.argv[1] + " is valid")
        return 0
    else:
        print("[*] File: " + sys.argv[1] + " is invalid")
        return -2
    
    
    
    



if __name__== "__main__":
    main()