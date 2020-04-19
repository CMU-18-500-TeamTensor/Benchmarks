'''
This file is suppposed to emulate an FPGA board that is waiting for the
broadcast signal from the Data Source Machine. Upon seeing the request, the 
server will send a response, signifiying that is has an available board
'''

import socket
import pickle



def read_content(IP,PORT):
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

    
    print("Listening for expected model size")
    sizeOfInt = 24
    data, address = sock.recvfrom(sizeOfInt)
    myModelSize = pickle.loads(data)
    print("What is my model size: ", myModelSize)
    chunk_size = 4096
    counter = 0
    finalPacket = []
    while(counter < myModelSize):
        print("Listening for packets about model info")
        data, address = sock.recvfrom(chunk_size)
        lower_bound = counter
        upper_bound = min(counter+chunk_size,myModelSize)
        print(lower_bound, upper_bound)
        packet = pickle.loads(data)
        finalPacket.extends(packet)
        print("Read this many bytes so far: ", counter)
        counter += chunk_size

    '''
    #print("Received model info of len:" , len(myModel))
    #chunk read for the model size
    while(readingModel):
        data, address = sock.recvfrom(4096)
        packet = pickle.loads(data)

        boardConfirmation = True
        if(boardConfirmation)
    '''
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