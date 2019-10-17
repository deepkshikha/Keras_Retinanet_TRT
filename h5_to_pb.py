from keras_retinanet.models import load_model, convert_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorflow.python.compiler.tensorrt import trt_convert as trt
#import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

model = load_model('resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
        
from tensorflow.keras import backend as K
frozen_graph = freeze_session(K.get_session(),
                              output_names=['filtered_detections/map/TensorArrayStack/TensorArrayGatherV3', 
                    'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3', 
                    'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3'])
                    
tf.train.write_graph(frozen_graph, "TRT2", "TF_model.pb", as_text=False)                    