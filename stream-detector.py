#!/usr/bin/python3

import os
import cv2
import sys
import numpy
import ntpath
import argparse

import mvnc.mvncapi as mvnc
import numpy as np

from utils import visualize_output
from utils import deserialize_output
from utils import mosquitto
from utils import stream_reader
from utils import camera_config as config

URL = config.camera_url
CONFIDENCE_THRESHOLD = config.confidence_threshold
GRAPH_FILE = config.graph
LABEL_FILE = config.labels

DIMENSIONS = tuple([300, 300])
MEAN = [127.5, 127.5, 127.5]
SCALE = 0.00789


# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_movidius():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.OpenDevice()

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( movidius ):

    # Read the graph file into a buffer
    with open(GRAPH_FILE, mode='rb') as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = movidius.AllocateGraph(blob)

    return graph

# ---- Step 3: Pre-process the images ----------------------------------------

def pre_process_image( frame ):

    img = cv2.resize(frame, DIMENSIONS).astype( numpy.float16 )
    img = ( img - numpy.float16(MEAN)) * SCALE

    return img

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image(movidius, graph, img, frame, labels ):

    # Load the image as a half-precision floating point array
    graph.LoadTensor( img, 'user object' )

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Get execution time
    inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

    # Deserialize the output into a python dictionary
    output_dict = deserialize_output.ssd( 
                      output, 
                      CONFIDENCE_THRESHOLD, 
                      frame.shape )
    
    # Print the results (each image/frame may have multiple objects)
    num_detections = output_dict['num_detections']
    print( "%i objects identified in %.1f ms" % (num_detections, numpy.sum( inference_time ) ) )

    for i in range( 0, num_detections):

        box = output_dict['detection_boxes_%i' % i]
        score = output_dict['detection_scores_%i' % i]
        claz = output_dict['detection_classes_%i' % i]
        name = labels[claz]

        (y1, x1) = box[0]
        (y2, x2) = box[1]

        mosquitto.publish(name, URL)
        
        display_str = ('{}: {}%'.format(name, score ))
        frame = visualize_output.draw_bounding_box( 
               y1, x1, y2, x2, 
               frame,
               thickness=1,
               color=(255, 255, 0),
               display_str=display_str )

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        
        cv2.imshow( 'NCS live inference', frame )
        if( cv2.waitKey(5) & 0xFF == ord('q')):
            dispose( movidius, graph )

# ---- Step 5: Unload the graph and close the device -------------------------

def dispose(movidius, graph ):
    
    graph.DeallocateGraph()
    movidius.CloseDevice()
    cv2.destroyAllWindows()
    mosquitto.disconnect()

# ---- Callback for the image stream --------------------------

def process_image(jpg, movidius, graph, labels) :
    
    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = pre_process_image(frame)
    infer_image(movidius, graph, img, frame, labels)
    
# ---- Main function (entry point for this script ) --------------------------
def main():

    movidius = open_movidius()
    print("Device ", movidius)
    graph = load_graph(movidius)
    print("Graph loaded")
    cleanup = lambda line: line.rstrip('/n').split(':')[1].strip()
    labels =[ cleanup(line) for line in
              open(LABEL_FILE) if line != 'classes\n']
    print("Labels loaded")
    
    callback = lambda jpg : process_image(jpg, movidius, graph, labels)
    
    mosquitto.connect()
    
    stream_reader.get_frames(URL, callback)


# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':
    main()


