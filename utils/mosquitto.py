#!/usr/bin/python3
'''
	MQTT Publisher for object recognition
'''

import paho.mqtt.client as mqtt
from utils import mqtt_config as config

# Edit the config file to use your own settings
broker = config.broker
port = config.port
topic = config.topic
username = config.username
password = config.password

client=mqtt.Client()

def connect():	
	
	client.username_pw_set(username, password)
	client.connect(broker, port, 60)
	print("Mosquitto connected to {}, topic {}".format(broker, topic) )
	client.loop_start()
	client.publish(topic, "Camera connected")

def publish(subtopic, msg):

	client.publish('{}/{}'.format(topic, subtopic), msg)
	# print('{}/{}: {}'.format(topic, subtopic, msg))


def disconnect():
	
	client.disconnect()
