import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np
#NOTE: idk what this does
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result = summary_string(model, input_size, batch_size, device, dtypes)
    return result


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''
    special_layer_list = []

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            #Also does conversion from tensor to array
            if("Linear" in class_name):
                #print("(Is this the numpy array: ", module.weight.cpu().data.numpy())
                special_layer_list.append((module.weight.cpu().data.numpy(),module.bias.cpu().data.numpy()))
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    
    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    
    total_params = 0
    total_output = 0
    trainable_params = 0
    layer_list = []
    for layer in summary:
        layer_list.append(layer)
        
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"
        

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_input_size_bytes = abs(np.prod(sum(input_size,())) * 4.)
    total_output_size_bytes = abs(2. * total_output * 4.)
    total_params_size_bytes = abs(total_params * 4.)
    print("Total input size (Bytes): ", total_input_size_bytes)
    print("Total output size (Bytes): ", total_output_size_bytes)
    print("Total params size (Bytes): ", total_params_size_bytes.item())
    total_size_bytes = total_input_size_bytes + total_output_size_bytes + total_params_size_bytes.item()
    print("Total size (Bytes): ", total_size_bytes)
    '''
    print(type(total_params.item()))
    print("Params size (MB): %0.2f" % total_params)
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    #total_size_bytes = abs((np.prod(sum(input_size, ())) * batch_size)) + abs(2*total_output) + total_params
    #print("Total size in bytes: ", total_size_bytes.item())
    total_size = total_params_size + total_output_size + total_input_size

    
    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    # return summary
    '''
    #print("What is len of special_layer_list: ", len(special_layer_list[0][0]))
    #kernelArray = special_layer_list[0][0]
    #biasArray = special_layer_list[0][1]
    #print("What is kernelArray: ", kernelArray)
    #print("What is biasArray: ", biasArray)
    #print("what is special_layer_list[0][0]: ", special_layer_list[0][1])
    #print("what is special_layer_list[0][1]: ", special_layer_list[0][1])
    return layer_list[:-1],special_layer_list, total_size_bytes