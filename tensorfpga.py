
import socket
import pickle
import sys
import numpy
import time
import torch
import struct

from modeldetailshelper import summary


'''
Each worker will be assigned a pipeline,
everything from a specific pipeline will be added to the 
worker's model list
'''
class Worker:
    name = "Worker Name 1"
    #worker_ip = 'pi.tensor.tk'
    worker_ip = 'localhost'
    worker_port = 18500
    data_pipeline = ""
    model_list = [] #Holds all models currently assigned to it
    def __init__(self, name, ip, port):
        self.name = name
        self.worker_ip = ip
        self.port = port

    def assign_pipeline(self,pipeline):
        self.data_pipeline = pipeline

    def add_model(self,model):
        self.model_list.append(model)

    
    def get_pipeline(self):
        return self.data_pipeline

    def get_model_list(self):
        return self.model_list

    def get_worker_name(self):
        return self.name



class DataPipelineManager:
    boards = 0 #total boards
    pipelines = 0 #total pipelines
    worker_list = [] #list of all workers
    pipeline_list = [] #list of all pipelines
    def __init__(self,boards):
        self.boards = boards

    def add_worker(self,name):
        new_worker = Worker(name,'localhost',18500)
        self.worker_list.append(new_worker)

    def get_worker(self,name):
        for worker in self.worker_list:
            if(worker.get_worker_name() == name):
                return worker
        print("Found no worker with this name")
        return

    #Adds a pipeline to a worker
    def add_pipeline(self, dp, pipeline_name, buffer_size):
        self.pipelines+=1
        m_list = []
        self.pipeline_list.append((pipeline_name,dp,m_list))
        return self.pipelines-1

    def add_model(self, model, dp_id):
        self.pipeline_list[dp_id][2].append(model)

    def add_pipeline_to_worker(self,pipeline_id,worker_index):
        self.worker_list[0].assign_pipeline("full_color")
        self.worker_list[0].model_list = self.pipeline_list[0][2]

    def get_pipeline(self, pipeline_id):
        return self.pipeline_list[pipeline_id][2]

    def get_model_from_pipeline(self, pipeline_id, model_num):
        return self.pipeline_list[pipeline_id][2][model_num]

    def get_model(self,worker_id,model_num):
        return self.worker_list[worker_id].model_list[model_num]

    def printNumBoards(self):
        print("Current connected boards: ", self.boards)

    def print_pipelines(self):
        #print("What are the pipelines", self.pipeline_list)
        for i in range(len(self.pipeline_list)):
            pipeline = self.pipeline_list[i]
            print("What is in pipeline: ", i, self.pipeline_list[i][2])

    def print_boards(self):
        print("Printing board list: ", self.worker_list)

    def print_worker_list(self, index):
        print("Printing model list for specific worker", self.worker_list[index].get_model_list())




'''
# Given a det of data pipelines and connectivity to a worker finder, produce
# an allocation of boards to data pipelines.
# This algorithm works as follows:
#  - Calculate the proportion of models assigned to each data pipeline
#  - Calculate the proportion of model managers assigned to each board
#  - For each data pipeline:
#     - Assign boards to it until the proportion of model managers assigned to
#     - is greater than or equal to the proportion of models that belongs to it
def allocate_boards(dpm):
    # Calculate the proportion of models that belong to each data pipeline
    num_models = 0
    for dp in dpm:
        num_models += len(dp.models)

    for dp in dpm:
        dp.prop_models = len(dp.models) / num_models

    # Calculate the proportion of model managers assigned to each board
    boards = WorkerFinder.get_boards()
    num_model_managers = 0
    for board in boards:
        num_model_managers.append(board.num_model_managers)

    for board in boards:
        board.prop_mm = board.num_model_managers / num_model_managers

    # For each data pipeline, assign boards to it until it has a proportionate
    # amount of computing power in the system.
    for dp in dpm:
        dp.prop_mm = 0.
        dp.boards = []
        while dp.prop_mm < dp.prop_models:
            board = boards[0]
            board.assign_dp(dp)
            dp.boards.append(board)
            dp.prop_mm += board.prop_mm

    # At this point, each board has been assigned one data pipeline and data
    # pipelines are distributed equitably across boards



# Given a Data Pipeline Manager, find the available boards and train all of
# the models on those boards.
def train_modells(dpm, data):

    # For each data pipeline
    for dp in dpm:
        # dp has already been assigned to a list of boards
        boards = dp.boards

        unassigned_models = dp.models

        # Sort the models from largest to smallest memory footprint
        unassigned_models.sort(decreasing=True)

        while unassigned_models != []:

            # Define no_models_assigned with dummy value that will cause
            # while loop to enter body
            no_models_assigned = False
            while not no_models_assigned:
                no_models_assigned = True
                for i in range(len(unassigned_models)):
                    model = unassigned_models[i]

                    # Equitably allocate work between boards
                    for board in boards:
                        if board.fits(model):
                            # Transfer the initial weights to the board
                            board.assign(model)

                            unassigned_models.remove(i)
                            no_models_assigned = False

            # The given board is now storing as many models as it can

            # Train all of the models that have been assigned to a board
            for i, (x, y) in data:
                processed_data = dp.pipeline_fn(x)

                dp.train(processed_data, y)
                # Later, this will store values in local statistics
                # structand return the struct to the user
                if i % 2000 == 0:
                    loss = dp.retrieve('loss')
                    print(loss)
'''

#returns the float32 version of any number (4 bytes)
def f32(float_num):
    return numpy.float32(float_num)

def now():
    return int(round(time.time() * 1000))

def get_true_output(x):
    y = torch.zeros(10)

    y[0] = x[0]
    y[1] = x[0]**2 + x[1]
    y[2] = x[2]
    y[3] = x[2] * x[3]
    y[4] = x[3]**2 * x[4]
    y[5] = x[5]
    y[6] = y[6]**3
    y[7] = y[7]
    y[8] = torch.sqrt(y[8])
    y[9] = y[9] if y[9] > 0 else 0.0
    return y


#Using UPD, send out broadcast to detect what boards are available
#Returns a lit of available boards

def get_boards(IP, PORT):
    '''
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    board_1 = (IP, PORT)
    message = "Is board 1 available"
    temp_list = ["this","is","my","message"]
    message1 = pickle.dumps(message)


    # Send data
    print("Sending out msg: ", message)
    sent = sock.sendto(message1, board_1)

    # Receive response
    print("Waiting to receive msg from server")
    while True:
        data, server = sock.recvfrom(4096)
        confirmation = pickle.loads(data)
        if(data):
            print("Received message from server: ", data)
            break;
    #NOTE: Currently assumes that there is no issues with reading what boards are 
    #available, automatically returns 4 connected boards
    '''
    return ["DE0-Nano_1", "DE0-Nano_2", "DE0-Nano_3", "DE0-Nano_4"]
    

#given a list, sends the list in the form of a bytes stream using TCP to the RaspPi
#does not return until receives confirmation from RaspPi
#Treating the data source machine as a client
#NOTE: THIS IS NOT EFFICIENT, TOO MANY BACK AND FORTHS, CAN REDUCE THE # OF CALLS
def send_content(content_list, worker):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #board = (worker.worker_ip, worker.worker_port)
    tcp_socket.connect(("localhost",18500))
    content_len = len(content_list)
    #print("content_len: ", content_len)
    #send total total elements, as well as size of total byte stream
    total_elements = f'{len(content_list):<20}'
    total_stream_size = f'{len(pickle.dumps(content_list)):<20}'
    
    #print(total_elements, total_stream_size)
    tcp_socket.send(bytes(total_elements, "utf-8"))
    tcp_socket.send(bytes(total_stream_size, "utf-8"))
    counter = 0
    #convert part of list to bytes stream, send small byte stream
    #convert back to list, add to final list
    print("Total_elements, stream size: ", total_elements, total_stream_size)
    while (counter < content_len):
        lower_bound = counter
        upper_bound = min(counter+1000, content_len)
        #print(lower_bound, upper_bound)
        model_fragment = content_list[lower_bound:upper_bound]
        #print("model_fragment: ", model_fragment)
        content_pickle = pickle.dumps(model_fragment)
        msgsize = f'{len(content_pickle):<20}'
        #send how many bytes are going to be sent first
        value = tcp_socket.send(bytes(msgsize, "utf-8"))
        #send the fragment
        value = tcp_socket.send(content_pickle)
        counter = upper_bound
    print("Finished sending content")
    tcp_socket.close()
    return 


#similar to send_content, but sends a str over instead of a list
def send_content_str(content_str, worker):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #board = (worker.worker_ip, worker.worker_port)
    tcp_socket.connect(("localhost",18500))
    msg_len = len(content_str)
    #content_len = len(content_list)
    #print("content_len: ", content_len)
    #send total total elements, as well as size of total byte stream
    #total_elements = f'{len(content_list):<20}'
    #total_stream_size = f'{len(pickle.dumps(content_list)):<20}'
    
    #print(total_elements, total_stream_size)
    total_str_len = f'{len(content_str):<20}'
    print("Sending msg_len: ", total_str_len)
    tcp_socket.send(bytes(total_str_len, "utf-8")) #20 byte message
    #tcp_socket.send(bytes(total_stream_size, "utf-8")) #20 byte message
    counter = 0
    #convert part of list to bytes stream, send small byte stream
    #convert back to list, add to final list
    #print("Total_elements, stream size: ", total_elements, total_stream_size)
    while (counter < msg_len):
        lower_bound = counter
        upper_bound = min(counter+1000, msg_len)
        #print(lower_bound, upper_bound)
        model_fragment = content_str[lower_bound:upper_bound]
        fragment_size = len(content_str)
        small_msg = model_fragment.encode()
        #print("model_fragment: ", model_fragment)
        #content_pickle = pickle.dumps(model_fragment)
        #msgsize = f'{len(content_pickle):<20}'
        #send how many bytes are going to be sent first
        value = tcp_socket.send(small_msg)
        #print("How many bytes were sent: ", value)
        #send the fragment

        #value = tcp_socket.send(content_pickle)
        counter = upper_bound
    print("Total bytes sent: ", counter)
    print("Finished sending content")
    tcp_socket.close()
    return 



def int_to_hex(int):
    return str(hex(int))[2:].zfill(8)

def float_to_hex(float):
    return str(hex(struct.unpack('<I', struct.pack('<f', float))[0]))[2:]

def list_to_hex(packet_list):
    hex_string = ""
    hex_val = ""
    #print("what is float_to_hex of 5.0: ", float_to_hex(f32(5.0)))
    #print("what is int_to_hex of 5:", int_to_hex(numpy.uint32(5)))
    f32_type = type(f32(5.0))
    uint32_type = type(numpy.uint32(5))
    #print(f32_type, uint32_type)
    hex_val = ""

    for i in range(len(packet_list),0,-1):
        #print("what is packet_list[i]: ", packet_list[i-1])
        
        element = packet_list[i-1]
        element_type = type(element)
        #print(element, element_type)
        if(element_type == f32_type):
            hex_val = float_to_hex(element)
        if(element_type == uint32_type):
            hex_val = int_to_hex(element)
        #print("what is hex_val: ", hex_val)
        hex_string += hex_val
        
    #print(hex_string)
    return hex_string
    
    

def write_content_to_file(id, hex_string):
    filename = "model_" + str(id) + ".txt"
    print(filename)
    file = open(filename, "w")
    file.write(hex_string)
    file.close()




#Sends one model to the board specified
def send_model(model, worker, model_id):


    #Holds the model information
    packet_list = []
    
    #Do this function today
    layer_list, special_layer_list, t_model_size = get_model_layers(model, model_id)
    class_name = str(model.__class__).split(".")[-1].split("'")[0]
    print("what is model.__class__", str(model.__class__))
    print("What is class_name: ", class_name)
    
    packet_list.append(numpy.uint32(5))
    packet_list.append(numpy.uint32(0))
    #chagne this to actually use the model #
    packet_list.append(numpy.uint32(model_id))
    #change this to use the actual size of the model
    #print("what is int32 of model_size: ", numpy.uint32(t_model_size))
    packet_list.append(numpy.uint32(t_model_size)) #size of model in word count
    packet_list.append(numpy.uint32(len(layer_list))) #of layers in a model

    specialCounter = 0
    #print("What is len of packet_list: ", len(packet_list))
    for layer in layer_list:
        if(layer == "ReLU"):
            packet_list.append(numpy.uint32(3)) #op code for ReLU layer
            #packet_list.append(f32(3)) #op code for ReLU layer
        elif(layer == "Linear"):
            kernelTensor = special_layer_list[specialCounter][0]
            biasTensor = special_layer_list[specialCounter][1]
            specialCounter+=1
            packet_list.append(numpy.uint32(1)) #op code for Linear layer
            #packet_list.append(f32(1)) #op code for Linear layer
            #packet_str += "1,"
            serializedKernel = tensor_to_list(kernelTensor,2)
            serializedBias = tensor_to_list(biasTensor,1)
            #print("What is len of serializedKernel: ", len(serializedKernel))
            #print("What is len of serializedBias: ", len(serializedBias))
            packet_list.extend(serializedKernel)
            packet_list.extend(serializedBias)
    #print("What is len of packet_list: ", len(packet_list))
    myType = type(packet_list[0])
    #print(myType)
    

    print("Sending model to RaspPi")
    time_start = now()
    #print("What is packet_list: ", packet_list)
    string_bytes = list_to_hex(packet_list)
    write_content_to_file(model_id, string_bytes)
    #print("what is string_bytes: ", string_bytes)
    #send_content(packet_list, worker)
    #send_content_str(string_bytes, worker)
    time_finish = now()
    print("Time to send model: ", time_finish-time_start)
    print("Finished sending the model")




#NOTE: sending data takes very long time, timing the whole thing is going to be tricky
def send_data(trainloader, worker):
    #NOTE:Should trainloader be defined here or should this be done by the user
    #NOTE:Will the labels always be the same size or will they be different
    #data_str = ""
    data_list = []
    counter = 0
    #time_start = now()
    data_list.append(numpy.uint32(11)) # opcode for data packet
    data_list.append(numpy.uint32(1)) # number of batches, only send one at a time
    for i, data in enumerate(trainloader, 0):
        print("New data batch: ", i)
        if (counter == 0):
            #inputs, labels = data
            inputs = torch.rand(20)
            labels = get_true_output(inputs)
            #print(type(inputs))
            input_list = inputs.numpy().tolist()
            serializedInput = tensor_to_list(input_list,1)
            #print(type(serializedInput[0]))
            data_list.extend(serializedInput)
            #convert from tensor to list
            label_list = labels.numpy().tolist()
            serializedLabel = tensor_to_list(label_list,1)
            data_list.extend(serializedLabel)
            #print(data_list)
            print("Sending data to RaspPi")
            time_start = now()
            send_content(data_list,worker)
            time_finish = now()
            print("Finished sending data")
            total_time = time_finish-time_start
            average_time = total_time/50000.0
            print(total_time, average_time)
            break
    #time_finish = now()
    print("Sent all batches of data")
    total_time = time_finish-time_start
    average_time = total_time/50000.0
    print(total_time, average_time)


#NOTE: This is kinda ugly, should clean this up, make it more efficient
def tensor_to_string(tensor, maxDim):
    #print("Starting tensor_serialization")
    #Resulting serialization, one dimension linear
    tensorStr = ""
    tensorStr += (str(maxDim) + ",")
    tensorStr += (str(len(tensor)) + ",")
    if(maxDim == 1):
        for i in range(len(tensor)):
            tensorStr += (str(tensor[i]) + ",")        

    elif (maxDim == 2):
        tensorStr += (str(len(tensor[0])) + ",")
        for j in range(len(tensor[0])):
            for i in range(len(tensor)):
                tensorStr += (str(tensor[i][j]) + ",")

    elif (maxDim == 3):
        tensorStr += (str(len(tensor[0])) + ",")
        tensorStr += (str(len(tensor[0][0])) + ",")

        for k in range(len(tensor[0][0])):
            for j in range(len(tensor[0])):
                for i in range(len(tensor)):
                    tensorStr += (str(tensor[i][j][k]) + ",")
    
    elif (maxDim == 4):
        tensorStr += (str(len(tensor[0])) + ",")
        tensorStr += (str(len(tensor[0][0])) + ",")
        tensorStr += (str(len(tensor[0][0][0])) + ",")

        for l in range(len(tensor[0][0][0])):
            for k in range(len(tensor[0][0])):
                for j in range(len(tensor[0])):
                    for i in range(len(tensor)):
                        tensorStr += (str((tensor[i][j][k][l])) + ",")
    
    return tensorStr


#NOTE: This returns a list instead of a string
def tensor_to_list(tensor, maxDim):
    #Resulting serialization, one dimension linear
    tensorList = []
    '''
    tensorList.append(f32(maxDim))
    tensorList.append(f32(len(tensor)))
    '''
    tensorList.append(numpy.uint32(maxDim))
    tensorList.append(numpy.uint32(len(tensor)))

    if(maxDim == 1):
        for i in range(len(tensor)):
            tensorList.append(numpy.float32(tensor[i]))
        

    elif (maxDim == 2):
        #tensorList.append(f32(len(tensor[0])))
        tensorList.append(numpy.uint32(len(tensor[0])))
        for j in range(len(tensor[0])):
            for i in range(len(tensor)):
                tensorList.append(numpy.float32(tensor[i][j]))

    elif (maxDim == 3):
        #tensorList.append(f32(len(tensor[0])))
        #tensorList.append(f32(len(tensor[0][0])))
        tensorList.append(numpy.uint32(len(tensor[0])))
        tensorList.append(numpy.uint32(len(tensor[0][0])))
        for k in range(len(tensor[0][0])):
            for j in range(len(tensor[0])):
                for i in range(len(tensor)):
                    tensorList.append(numpy.float32(tensor[i][j][k]))
    
    elif (maxDim == 4):
        #tensorList.append(f32(len(tensor[0])))
        #tensorList.append(f32(len(tensor[0][0])))
        #tensorList.append(f32(len(tensor[0][0][0])))
        tensorList.append(numpy.uint32(len(tensor[0])))
        tensorList.append(numpy.uint32(len(tensor[0][0])))
        tensorList.append(numpy.uint32(len(tensor[0][0][0])))
        for l in range(len(tensor[0][0][0])):
            for k in range(len(tensor[0][0])):
                for j in range(len(tensor[0])):
                    for i in range(len(tensor)):
                        tensorList.append(numpy.float32(tensor[i][j][k][l]))
    
    return tensorList



#Given a model, returns a list of all of the layers in the order that they appear
def get_model_layers(model, model_id):
    #layer lists holds all the layers
    #special_layer_list holds all the information for each Linear layer
    if(model_id < 15):
        layer_list, special_layer_list, t_model_size = summary(model,(1,1,20))
    else:
        layer_list, special_layer_list, t_model_size = summary(model,(1,1,10)) 
    loop_length = len(layer_list)
    #Remove the extra values after the layer list name
    for i in range(loop_length):
        element = layer_list.pop(0)
        index = len(element) - element.index("-")
        element = element[:-index]
        layer_list.append(element)
    print("What is t_model_size: ", t_model_size)
    return layer_list, special_layer_list, t_model_size

def train_models(dpm,trainloader):
    IP = "localhost"
    PORT = 18500

    #Get all available boards and add to board list
    board_list = get_boards(IP,PORT)
    print("what is board_list:", board_list)
    for i in range (len(board_list)):
        dpm.add_worker(board_list[i])
    
    #dpm.print_boards()
    #dpm.print_pipelines()

    #dpm.print_worker_list(0)
    print(dpm.get_pipeline(0))
    worker1 = dpm.get_worker("DE0-Nano_1")
    worker2 = dpm.get_worker("DE0-Nano_2")
    #NOTE: Hardcoded to use 15 since thats how many models per dataset
    total_models = len(dpm.get_pipeline(0)) + len(dpm.get_pipeline(1))
    for i in range(total_models):
        print("which pipeline: ", i//15)
        model = dpm.get_model_from_pipeline(i//15,i%15)
        print("my model at i: ", i, model)
        mymodel = model().to(0)
        send_model(mymodel,worker1, i)
        time.sleep(1)
    #first parameter is worker #, second # is index of model
    #After all the models are sent, send the data
    send_data(trainloader, worker1)




def main():
    IP = "localhost"
    PORT = 18500
    '''
    my1DTensor = [1,2,3]

    my2DTensor = [[1,2],[3,4],[5,6]]

    my3DTensor = [[[1,2],[3,4],[5,6]],
                [[7,8],[9,10],[11,12]],
                [[13,14],[15,16],[17,18]],
                [[19,20],[21,22],[23,24]]]
    
    my4DTensor = [[[[1,2],[3,4],[5,6]],
                [[7,8],[9,10],[11,12]],
                [[13,14],[15,16],[17,18]],
                [[19,20],[21,22],[23,24]]],
                [[[25,26],[27,28],[29,30]],
                [[31,32],[33,34],[35,36]],
                [[37,38],[39,40],[41,42]],
                [[43,44],[45,46],[47,48]]]]

    tensorList = tensor_serializer(my4DTensor, 4)
    tensorStr = tensor_serializer2(my4DTensor, 4)
    #print(tensorList)
    #print(tensorStr)
    tensorList2 = list(tensorStr.split(","))
    tensorList2 = tensorList2[:-1]
    for i in range(max(len(tensorList2),len(tensorList))):
        print(tensorList[i],tensorList2[i])
    '''

    #get_boards(IP, PORT)
    packet_list = []
    for i in range(10):
        packet_list.append(numpy.uint32(i))        
    #print("what is packet_list len: ", len(packet_list))
    print("What is packet_list: ", packet_list)
    '''
    worker = Worker("Board1", "localhost", 18500)
    send_content(packet_list, worker)
    '''
    list_to_hex(packet_list)



if __name__ == '__main__':
    main()
