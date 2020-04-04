'''
This file is suppposed to emulate an FPGA board that is waiting for the
broadcast signal from the Data Source Machine. Upon seeing the request, the 
server will send a response, signifiying that is has an available board
'''

import socket



def read_model(IP,PORT):
    return "implement me"


#Hosts server, waits for incoming call
def main(IP, PORT):
    responseGood = "Board #1 is available"
    responseBad = "Board #1 is not available"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    #Set up server on board side
    board_1 = (IP, PORT)
    sock.bind(board_1)

    print('starting up on {} port {}'.format(*board_1))

    while True:
        print("Listening for message")
        data, address = sock.recvfrom(4096)

        print("Received broadcast: ", data)

        sock.sendto(responseGood.encode(),address)
        print("Sent response: ", responseGood)
        break;
    print("Exiting out of server")
    



if __name__ == '__main__':
    IP = "localhost"
    PORT = 18500
    main(IP,PORT)