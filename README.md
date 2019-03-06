# Movidius Remote Object Detection
Object recognition on a remote video stream, for a Raspberry Pi using a 
[Movidius Neural Compute Stick](https://software.intel.com/en-us/neural-compute-stick).

A (not necessarily) separate Pi runs [motioneye](https://randomnerdtutorials.com/install-motioneyeos-on-raspberry-pi-surveillance-camera-system/) 
software, and this provides the remote video stream. The idea is that you do not have to have all your hardware on one device.

If you do not have a Movidius NCS device, [this project](https://github.com/Turlough/Object-Detection-From-Remote-Camera)
can be run on a PC instead.

# Features
An MQTT message is published for each object detected in each frame. The topic is 'camera/*label*' (where label is the label of each object detected)
and the message is the URL of the source video. 

A [MQTT Pushbullet Notification relay](https://github.com/Turlough/MQTT-Pushbullet-Notification-Relay) subscribes to the 
'camera/person' topic. When a new person is detected in the video stream, a Pushbullet notification is sent to your phone.

In my home network, the Camera Pi, Movidius Pi, Pushbullet Relay Pi, and MQTT Broker are on separate devices. 
However, you could run them all on the same device.
