'''
This file is suppposed to emulate an FPGA board that is waiting for the
broadcast signal from the Data Source Machine. Upon seeing the request, the 
server will send a response, signifiying that is has an available board
'''

import socket



def read_content(IP,PORT):
    responseGood = "Board #1 is available"
    responseBad = "Board #1 is not available"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    #Set up server on board side
    board_1 = (IP, PORT)
    sock.bind(board_1)

    print('starting up on {} port {}'.format(*board_1))
    
    print("Listening for message")
    data, address = sock.recvfrom(4096)

    print("Received broadcast: ", data)

    sock.sendto(responseGood.encode(),address)
    print("Sent response: ", responseGood)

    print("Listening for packet")
    data, address = sock.recvfrom(4096)

    print("Received packet: ", data)

    '''
    print("Listening for batch")
    data, address = sock.recvfrom(4096)

    print("Received packet: ", data)
    '''
    

#Hosts server, waits for incoming call
def main(IP, PORT):
    read_content(IP,PORT)

    print("Server finished")
    



if __name__ == '__main__':
    IP = "localhost"
    PORT = 18500
    main(IP,PORT)