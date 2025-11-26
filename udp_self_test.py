# Code to test if data is being sent correctly through UDP from main.py

import numpy as np
import socket

HOST = "localhost"
PORT = 65432 # Port to listen on

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    
    s.bind((HOST,PORT))
   
    while True:
        data_byte = s.recvfrom(1024)[0]
        if not data_byte:
            break
        data = np.frombuffer(data_byte, dtype = np.double, count = 2)
        print(data)
