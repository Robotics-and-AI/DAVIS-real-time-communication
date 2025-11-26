# Code to comunicate with the server to which the UDP data is sent in main.py, in order to evaluate total elapsed time

import numpy as np
import time
import os
import socket

HOST = "123.123.123.123" # server IP address to receive from
PORT = 65433 # Server port to receive from

time_list = []

try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        
        s.bind((HOST,PORT))

        start_time = time.time()
        while ((time.time() - start_time) <= 1000): # Timer for listening

            data = s.recvfrom(1024)[0]
            time_value = np.frombuffer(data, dtype = np.float64, count = 1) # Receive time from server
            time_list.append([time_value,time.time()])

            if not data:
                break

finally:
    csv_file = os.path.join(os.path.abspath(""), "time.txt")
    np.savetxt(csv_file, time_list, fmt = ["%.5f","%.5f"])
