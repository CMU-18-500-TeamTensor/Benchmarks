
import socket


class Worker:
    name = "Worker Name 1"
    data_pipeline = ""
    model_list = [] #Holds all models currently assigned to it
    def __init__(self, name):
        self.name = name

    def assign_pipeline(self,pipeline):
        self.data_pipeline = pipeline

    def add_model(self,model):
        self.model_list.append(model)

    def get_pipeline(self):
        return self.data_pipeline



class DataPipelineManager:
    boards = 0 #total boards
    pipelines = 0 #total pipelines
    worker_list = [] #list of all workers
    pipeline_list = [] #list of all pipelines
    def __init__(self,boards):
        self.boards = boards

    def add_worker(self,name):
        self.worker_list.append(name)

    #Adds a pipeline to a worker
    def add_pipeline(self, dp, pipeline_name, buffer_size):
        self.pipelines+=1
        m_list = []
        self.pipeline_list.append((pipeline_name,dp,m_list))
        return self.pipelines-1

    def add_model(self, model, dp_id):
        self.pipeline_list[dp_id][2].append(model)


    def printNumBoards(self):
        print("Current connected boards: ", self.boards)

    def print_pipelines(self):
        print("What are the pipelines", self.pipeline_list)

    def print_boards(self):
        print("what is the current worker list: ", self.worker_list)




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


'''
# Given a Data Pipeline Manager, find the available boards and train all of
# the models on those boards.
def train_models(dpm, data):

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

#Using UPD, send out broadcast to detect what boards are available
#Returns a lit of available boards
def get_boards(IP, PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    
    board_1 = (IP, PORT)
    message = "Is board 1 available"

    # Send data
    print("Sending out msg: ", message)
    sent = sock.sendto(message.encode(), board_1)

    # Receive response
    print("Waiting to receive msg from server")
    while True:
        data, server = sock.recvfrom(4096)
        if(data):
            print("Received message from server: ", data)
            break;

    print("Exiting out of client")
    return ["DE0-Nano_1"]
    

#Actually sending data over to a specific board
def send_model(board):
    return "hi"

def send_data(board):
    return "hello"

def train_models(dpm,data):
    IP = "localhost"
    PORT = 18500
    dpm.print_pipelines

    #Get all available boards and add to board list
    board_list = get_boards(IP,PORT)
    for i in range (len(board_list)):
        dpm.add_worker(board_list[i])
    
    dpm.print_boards()

    



#Note: can we assume that all of our tensors will be 4 dimensions?
#Note: Current only works for 4 dimension tensor
#Returns serialized tensor in list format
#This is kinda ugly, should clean this up, make it more efficient
def tensor_serializer(tensor, maxDim):
    print("Starting tensor_serialization")
    #Resulting serialization, one dimension linear
    tensorList = []
    tensorList.append(maxDim)
    tensorList.append(len(tensor))
    if(maxDim == 1):
        tensorList.append(len(tensor[0]))
        for j in range(len(tensor[0])):
            for i in range(len(tensor)):
                tensorList.append(tensor[i][j])
    elif (maxDim == 3):
        tensorList.append(len(tensor[0]))
        tensorList.append(len(tensor[0][0]))
        for k in range(len(tensor[0][0])):
            for j in range(len(tensor[0])):
                for i in range(len(tensor)):
                    tensorList.append(tensor[i][j][k])
    
    elif (maxDim == 4):
        tensorList.append(len(tensor[0]))
        tensorList.append(len(tensor[0][0]))
        tensorList.append(len(tensor[0][0][0]))
        for l in range(len(tensor[0][0][0])):
            for k in range(len(tensor[0][0])):
                for j in range(len(tensor[0])):
                    for i in range(len(tensor)):
                        tensorList.append(tensor[i][j][k][l])
    
    return tensorList

def main():
    IP = "localhost"
    PORT = 18500
    my1DTensor = [[1,2],[3,4],[5,6]]

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
    print("What is tensorList: ", tensorList)

    #get_boards(IP, PORT)


if __name__ == '__main__':
    main()
