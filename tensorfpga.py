
import socket
import pickle
import sys
import numpy

from modeldetailshelper import summary


'''
Each worker will be assigned a pipeline,
everything from a specific pipeline will be added to the 
worker's model list
'''
class Worker:
    name = "Worker Name 1"
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

    def add_pipeline_to_worker(self):
        self.worker_list[0].assign_pipeline("Full Color")
        self.worker_list[0].model_list = self.pipeline_list[0][2]

    def get_model(self,worker_id,model_num):
        return self.worker_list[worker_id].model_list[model_num]

    def printNumBoards(self):
        print("Current connected boards: ", self.boards)

    def print_pipelines(self):
        print("What are the pipelines", self.pipeline_list)

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
    

#Sends one model to the board specified
def send_model(model, worker):

    #Int tensor to use for verification purposes
    kernelTensor = [[[[1,2],[3,4],[5,6]],
                [[7,8],[9,10],[11,12]],
                [[13,14],[15,16],[17,18]],
                [[19,20],[21,22],[23,24]]],
                [[[25,26],[27,28],[29,30]],
                [[31,32],[33,34],[35,36]],
                [[37,38],[39,40],[41,42]],
                [[43,44],[45,46],[47,48]]]]

    linear_tensor = [[1,2],[3,4],[5,6]]

    biasTensor = [1,2,3]

    packet_list = []
    #Do this function today
    layer_list, special_layer_list = get_model_layers(model)
    print("Layer List: ", layer_list)
    packet_list.append(f32(5.0)) #opcode for 
    packet_list.append(f32(0.0)) #pipeline id
    packet_list.append(f32(14.0)) #model id
    packet_list.append(f32((len(layer_list)))) #number of layers

    specialCounter = 0
    #print("What is len of packet_list: ", len(packet_list))
    for layer in layer_list:
        if(layer == "ReLU"): 
            packet_list.append(f32(3)) #op code for ReLU layer
        elif(layer == "Linear"):
            kernelTensor = special_layer_list[specialCounter][0]
            biasTensor = special_layer_list[specialCounter][1]
            specialCounter+=1
            packet_list.append(f32(1)) #op code for Linear layer
            #packet_str += "1,"
            serializedKernel = tensor_to_list(kernelTensor,2)
            serializedBias = tensor_to_list(biasTensor,1)
            print("What is len of serializedKernel: ", len(serializedKernel))
            print("What is len of serializedBias: ", len(serializedBias))
            packet_list.extend(serializedKernel)
            packet_list.extend(serializedBias)

    print("What is len of packet_list: ", len(packet_list))
    myType = type(packet_list[0])
    print(myType)
    #make sure every element in packet_list is the same type "<class 'numpy.float32'>"
    for i in range(len(packet_list)):
        element = packet_list[i]
        if(type(element) != myType):
            print(type(element), myType, i)    
            exit(1)
    print("Looks good to me")
    
    #NOTE: COME BACK TO THIS, CHANGE THIS TO TCP    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    board_1 = (worker.worker_ip, worker.worker_port)
    
    # Sending size of model first
    msgsize = pickle.dumps(len(packet_list))
    #NOTE: MAKING TEMP PACKET_LIST FOR TESTING PURPOSES REMOVE FROM HERE
    packet_list = []
    for i in range(200):
        packet_list.append(f32(i))

    #NOTE: TO HERE WHEN FINISHED TESTING
    print("Sending out size of message: ", len(packet_list))
    sent = sock.sendto(msgsize, board_1)

    # Sending out model info in chunks of 10,000 bytes
    print("Sending out the actual msg now")
    chunk_size = 4096
    counter = 0
    sent = 0
    #CURRENT ISSUES: HAVING TROUBLE SENDING MODEL OVER, NEED TO FIGURE OUT HOW TO CHUNK SEND
    #SOLUTION: SWAP THIS TO USE TCP INSTEAD FOR CHUNK SENDING
    while(sent < len(packet_list)):
        lower_bound = counter
        upper_bound = min(counter+chunk_size,len(packet_list))
        print(lower_bound, upper_bound)
        message = pickle.dumps(packet_list[lower_bound:upper_bound])
        sent += sock.sendto(message,board_1)
        print("Sent this many bytes: ", sent)
        print("Send this many bytes so far: ", counter)
        counter += chunk_size
    print("Finished sending the model")




def send_data(trainloader, worker):
    #NOTE:Should trainloader be defined here or should this be done by the user
    #NOTE:Will the labels always be the same size or will they be different
    #data_str = ""
    data_list = []
    counter = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        #print(len(inputs), len(inputs[0]), len(inputs[0][0]), len(inputs[0][0][0]), numpy.float32((inputs[0][0][0][0])))
        #print(type(inputs))
        input_list = inputs.numpy()
        #print(type(input_list))
        #print(type(labels))
        serializedInput = tensor_to_list(input_list,4)
        print(type(serializedInput[0]))
        data_list.extend(serializedInput)
        #print("What is len of serialized", len(serializedInput))
        #print("What is serializedInput: ", serializedInput)
        #data_str += serializedInput
        counter+=1
    print("what is counter: ", counter)
    #print("Whats the final data_str: ", data_str)
    #print("What is len of final data_str: ", len(data_str))


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
    tensorList.append(f32(maxDim))
    tensorList.append(f32(len(tensor)))
    if(maxDim == 1):
        for i in range(len(tensor)):
            tensorList.append(numpy.float32(tensor[i]))
        

    elif (maxDim == 2):
        tensorList.append(f32(len(tensor[0])))
        for j in range(len(tensor[0])):
            for i in range(len(tensor)):
                tensorList.append(numpy.float32(tensor[i][j]))

    elif (maxDim == 3):
        tensorList.append(f32(len(tensor[0])))
        tensorList.append(f32(len(tensor[0][0])))
        for k in range(len(tensor[0][0])):
            for j in range(len(tensor[0])):
                for i in range(len(tensor)):
                    tensorList.append(numpy.float32(tensor[i][j][k]))
    
    elif (maxDim == 4):
        tensorList.append(f32(len(tensor[0])))
        tensorList.append(f32(len(tensor[0][0])))
        tensorList.append(f32(len(tensor[0][0][0])))
        for l in range(len(tensor[0][0][0])):
            for k in range(len(tensor[0][0])):
                for j in range(len(tensor[0])):
                    for i in range(len(tensor)):
                        tensorList.append(numpy.float32(tensor[i][j][k][l]))
    
    return tensorList



#Given a model, returns a list of all of the layers in the order that they appear
def get_model_layers(model):
    #layer lists holds all the layers
    #special_layer_list holds all the information for each Linear layer
    layer_list, special_layer_list = summary(model,(3,32,32))
    loop_length = len(layer_list)
    for i in range(loop_length):
        element = layer_list.pop(0)
        index = len(element) - element.index("-")
        element = element[:-index]
        layer_list.append(element)
    return layer_list, special_layer_list

def train_models(dpm,trainloader):
    IP = "localhost"
    PORT = 18500

    #Get all available boards and add to board list
    board_list = get_boards(IP,PORT)
    print("what is board_list:", board_list)
    for i in range (len(board_list)):
        dpm.add_worker(board_list[i])
    
    dpm.print_boards()
    dpm.print_pipelines()
    dpm.add_pipeline_to_worker()
    dpm.print_worker_list(0)
    #first parameter is worker #, second # is index of model
    model = dpm.get_model(0,1)().to(0)
    print("Which model is this: ", model)
    #send_model should send in the model and the board OBJECT
    worker = dpm.get_worker("DE0-Nano_1")
    print("What is worker: ", worker)
    #send_model(model,worker)
    #After all the models are sent, send the data
    send_data(trainloader, worker)




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

    get_boards(IP, PORT)


if __name__ == '__main__':
    main()
