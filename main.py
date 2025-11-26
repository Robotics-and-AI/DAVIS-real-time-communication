from collections import deque
import numpy as np
import socket
import struct
import threading
import os
import time

class SafePrint():

    def __init__(self):
        self._lock = threading.Lock()

    # ------------ SAFE PRINTING METHOD ------------

    def safe_print(self, text:str) -> None:
        # To print while running a thread
        with self._lock:
            print(text)

    # ------------ COLORED TERMINAL TEXT ------------

    def print_success(self, text:str) -> None: # Green text
        self.safe_print("\033[92m {}\033[00m".format(text)) 

    def print_error(self, text:str) -> None: # Red text
        self.safe_print("\033[91m {}\033[00m" .format(text))


class ROI:

    def __init__(self, edge):

        self._EDGE = edge
    
    # ------------ REGION OF INTEREST METHOD ------------

    def get_center(self,x:np.ndarray,y:np.ndarray) -> list[list[int]]:

        x_sum = np.zeros(240, dtype = 'int32')
        y_sum = np.zeros(180, dtype = 'int32')

        # Compute the sum for each coordinate
        for i in range(len(x)):
            x_sum[x[i]] = x_sum[x[i]] + 1
            y_sum[y[i]] = y_sum[y[i]] + 1
        
        # Find the ROI
        x_range = self._get_range(x_sum)
        y_range = self._get_range(y_sum)

        x_size = x_range[1] - x_range[0]
        y_size = y_range[1] - y_range[0]

        # Make ROI square
        if x_size > 0 and y_size > 0:

            if x_size > y_size: # Landscape
                x_sum_roi = x_sum[x_range[0]:x_range[1]]
                sums_in_range = np.convolve(x_sum_roi,np.ones(y_size))
                sum_list = sums_in_range.tolist()
                offset = sum_list.index(max(sum_list)) - y_size
                x_range[0] = x_range[0] + offset
                x_range[1] = x_range[0] + y_size

            if y_size > x_size: # Portrait
                y_sum_roi = y_sum[y_range[0]:y_range[1]]
                sums_in_range = np.convolve(y_sum_roi,np.ones(x_size))
                sum_list = sums_in_range.tolist()
                offset = sum_list.index(max(sum_list)) - x_size
                y_range[0] = y_range[0] + offset
                y_range[1] = y_range[0] + x_size

            x_centre = (x_range[0] + x_range[1])//2
            y_centre = (y_range[0] + y_range[1])//2

            return [x_centre, y_centre]
        else:
            return [0,0]
        
    def _get_range(self, coord_sum:list[int]) -> list[int]:
        
        window = deque([])
        sum_window = 0
        edge_list = []
        
        # Verify where are the values of sums that are bigger than the average of the sums
        coord_idx = np.argwhere(coord_sum > np.mean(coord_sum))
        coord_idx = [element for sublist in coord_idx for element in sublist] # Flatten list
        diff = np.diff(coord_idx) # Get differences between sucessive values
        
        for i, val in enumerate(diff[:-self._EDGE]):
            window.append(val)
            sum_window += val
            if len(window) == self._EDGE:
                # If there are self._EDGE consecutive values bigger than the average of the sums, add index to edge_list
                if sum_window == self._EDGE:
                    edge_list.append(coord_idx[i])
                sum_window -= window.popleft()
        
        if len(edge_list) < 2:
            return [0,0]
        else:
            return [edge_list[0], edge_list[-1]]

class RealTime:

    def __init__(self, edge, buffersize, host, port, avg_param, min_event_threshold):

        self._lock = threading.Lock()
        self._algorithms = ROI(edge)
        self._safe_io = SafePrint()
        
        self._host = host
        self._port = port
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._serial_thread = threading.Thread(target = self._read_udp)
        self._serial_thread.daemon = True
        self._q = deque([])
        self.time_list = []
        
        self._BUFFERSIZE = buffersize
        self._exit_cue = False
        self._is_reading_udp = False
        self._THRESHOLD = min_event_threshold

        self._mov_avg_param = avg_param
        self._avg_q = deque([])
        self._mov_avg_x = 0
        self._mov_avg_y = 0

    # ------------ READ UDP (THREAD) METHOD ------------

    def send_roi(self) -> list:
        
        # Start UDP reading thread
        self._set_is_reading_udp(True)
        self._serial_thread.start()

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            counter = 0
            
            # Initialize moving average 
            while counter < self._mov_avg_param:
                if self._q:
                    [x,y] = self._get_from_queue()
                    x_centre, y_centre = self._algorithms.get_center(x,y)
                        
                    if x_centre != 0 and y_centre != 0:
                        self._add_to_avg_queue([x_centre,y_centre])
                        self._mov_avg_x += -(-x_centre//self._mov_avg_param)
                        self._mov_avg_y += -(-y_centre//self._mov_avg_param)
                        counter += 1
            
            # Actual start of sending data
            start_time = time.time()
            while ((time.time() - start_time) <= 1000): # Timer for listening

                if self._q:
                    
                    [x,y] = self._get_from_queue() # Get list of coordinate data (event packet) from queue
                    x_centre, y_centre = self._algorithms.get_center(x,y)
                    if x_centre != 0 and y_centre != 0:
                        
                        # Update moving average
                        [x_last,y_last] = self._get_from_avg_queue()
                        self._add_to_avg_queue([x_centre,y_centre])
                        self._mov_avg_x += -(-x_centre//self._mov_avg_param)
                        self._mov_avg_x -= -(-x_last//self._mov_avg_param)
                        self._mov_avg_y += -(-y_centre//self._mov_avg_param)
                        self._mov_avg_y -= -(-y_last//self._mov_avg_param)

                        # Writes the predicted ROI center using moving average and the current time
                        data = np.array([self._mov_avg_x, self._mov_avg_y,time.time()], dtype = np.float64)

                        data_bytes = data.tobytes()
                        s.sendto(data_bytes,(self._host,self._port)) # Send data to HOST

            return self.time_list
        
    # ------------ READ UDP (THREAD) METHOD ------------

    def _read_udp(self) -> None:
        # Receives data from jAER through UDP and adds it to queue

        self.udp.bind(("localhost",8991)) # Make sure this is defined in jAER

        while self._get_is_reading_udp():

            data = self.udp.recvfrom(self._BUFFERSIZE)[0] # Get data from jAER through UDP
            address = []

            event_data = struct.iter_unpack(">II",data)
            for event in event_data: 
                address.append(event[0])
            
            if address: #if not empty
                if len(address) > self._THRESHOLD: # Decode event data packet
                    address_np = np.array(address)
                    x = (address_np & 0x003ff000) >> 12
                    y = (address_np & 0x7fc00000) >> 22
                    self._add_to_queue([x,y]) # Sends list of coordinates to queue

    # ------------ SETTER AND GETTER METHODS ------------

    # - Set and get for variable is_reading_serial -
    def _set_is_reading_udp(self, val: bool) -> None:
        with self._lock:
            self._is_reading_udp = val
    def _get_is_reading_udp(self) -> bool:
        with self._lock:
            return self._is_reading_udp is True
        
    # - Handler of queue _q -
    def _add_to_queue(self, data) -> None:
        with self._lock:
            self._q.append(data)
    def _get_from_queue(self):
        with self._lock:
            return self._q.popleft()
        
    # - Handler of moving average queue avg_q -
    def _add_to_avg_queue(self, data) -> None:
        with self._lock:
            self._avg_q.append(data)

    def _get_from_avg_queue(self):
        with self._lock:
            return self._avg_q.popleft()

# -----------------------------------------
# -----------------------------------------
        
if __name__ == "__main__":
    
    # Parameters
    EDGE = 2
    BUFFERSIZE = 6000
    HOST = "123.123.123.123" # server IP address to send to - Change to "localhost" for udp_self_test.py
    PORT = 65432 # server port to send to 
    AVG_PARAM = 3
    THRESHOLD = 250

    real_time = RealTime(EDGE, BUFFERSIZE, HOST, PORT, AVG_PARAM, THRESHOLD)
    safe_print = SafePrint()

    try:
        time_list = real_time.send_roi()

        #csv_file = os.path.join(os.path.abspath(""), "coordinates.txt")
        #np.savetxt(csv_file, time_list, fmt = ["%d","%d","%0.5f"])

    except Exception as e:
        safe_print.print_error(str(e))

    safe_print.print_success("Program was terminated")

    
