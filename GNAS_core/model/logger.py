import os

logger_path = os.path.split(os.path.realpath(__file__))[0][:-(7+len('GNAS_core'))] + "/logger"


def gnn_architecture_performance_save(gnn_architecture, performance, data_save_name):
    if not os.path.exists(logger_path + '/' + str(data_save_name)):
        os.makedirs(logger_path + '/' + str(data_save_name))

    with open(logger_path + '/' + str(data_save_name) + "/" + str(data_save_name) + "_gnn_logger.txt", "a+") as f:
        f.write(str(gnn_architecture) + ":" + str(performance) + "\n")

    print("gnn architecture and performance save")
    print("save path: ", logger_path + '/' + str(data_save_name) + "/" + str(data_save_name) + "_gnn_logger.txt")
    print(50 * "=")



def gnn_architecture_performance_load(data_save_name):
    gnn_architecture_list = []
    performance_list = []

    with open(logger_path + '/' + str(data_save_name) + '/' + str(data_save_name) + "_gnn_logger.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            line = line.split(":")
            gnn_architecture = eval(line[0])
            gnn_architecture_list.append(gnn_architecture)
            performance = eval(line[1].replace("\n", ""))
            performance_list.append(performance)

    return gnn_architecture_list, performance_list
