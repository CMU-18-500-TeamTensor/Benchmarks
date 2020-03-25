
import socket



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


#Using UPD, send out broadcast to detect what boards are available
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
    

#Actually sending data over to a specific board
def send_data(board):
    return "hi"

def main():
    IP = "localhost"
    PORT = 18500
    get_boards(IP, PORT)


if __name__ == '__main__':
    main()
