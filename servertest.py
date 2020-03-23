'''
This file is suppposed to emulate an FPGA board that is waiting for the
broadcast signal from the Data Source Machine. Upon seeing the request, the 
server will send a response, signifiying that is has an available board
'''

import socket



#Hosts server, waits for incoming call
def main():
    responseGood = "Board #1 is available"
    responseBad = "Board #1 is not available"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    server_address = ("localhost", 10000)
    print('starting up on {} port {}'.format(*server_address))
    sock.bind(server_address)

    while True:
        print("Listening for message")
        data, address = sock.recvfrom(4096)

        print("Received broadcast: ", data)

        sock.sendto(responseGood.encode(),server_address)
        print("Sent response: ", responseGood)
        break;
    print("Exiting out of server")
    



if __name__ == '__main__':
    main()