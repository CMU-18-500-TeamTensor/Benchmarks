'''
This file is suppposed to emulate an FPGA board that is waiting for the
broadcast signal from the Data Source Machine. Upon seeing the request, the 
server will send a response, signifiying that is has an available board
'''

import socket
import pickle


def get_boards(IP, PORT):
    responseGood = pickle.dumps("Board #1 is available")
    responseBad = pickle.dumps("Board #1 is not available")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    #Set up server on board side
    board_1 = (IP, PORT)
    sock.bind(board_1)
    print('starting up on {} port {}'.format(*board_1))
    print("Listening for message")
    data, address = sock.recvfrom(4096)
    message = pickle.loads(data)

    print("Received broadcast: ", message)

    sock.sendto(responseGood,address)
    print("Sent response: ", responseGood)

    sock.close()
    return


def read_content(IP,PORT):
    #TCP FROM THIS POINT FORWARD
    board_1 = (IP, PORT)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.bind(("localhost",18500))
    tcp_socket.listen(5)

    
    clientsocket, address = tcp_socket.accept()
    print(f"Connection from {address} has been established")
    msg1 = clientsocket.recv(20)
    msg2 = clientsocket.recv(20)
    total_elements = int(msg1.decode("utf-8"))
    total_stream_size = int(msg2.decode("utf-8"))

    print("Total_elements, stream size: ", total_elements, total_stream_size)
    counter = 0
    
    model_msg = []
    while(counter < total_elements):
        lower_bound = counter
        upper_bound = min(counter+1000, total_elements)
        #print(lower_bound, upper_bound)
        
        bytes_to_read = int(clientsocket.recv(20).decode("utf-8"))
        model_pickle = clientsocket.recv(bytes_to_read)
        model_fragment = pickle.loads(model_pickle)
        #print("Model_fragment: ", model_fragment)
        model_msg.extend(model_fragment)
        counter = upper_bound

    #print("looks goood to me")
    tcp_socket.close()
    return
    

#Hosts server, waits for incoming call
def main(IP, PORT):
    print("Attempting to retrieve boards")
    #get_boards(IP, PORT)
    print("Boards retrieved, attempting to get model info")
    #called 30 times to get each model
    for i in range(30):
        read_content(IP,PORT)
    print("Model info retrieved, attempting to get data")
    #called another time to get the data
    #for i in range(50000):
    read_content(IP,PORT)
    print("Data retrieved")
    print("Server finished")
    

if __name__ == '__main__':
    IP = "localhost"
    PORT = 18500
    main(IP,PORT)