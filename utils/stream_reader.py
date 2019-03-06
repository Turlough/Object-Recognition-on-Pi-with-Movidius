#!/usr/bin/python3
'''
	Utility class to read jpgs from an mpeg stream.
	Usage: Call get_frames, passing it the url of the stream to process,
	and a callback for however you want to process each returned jpg.
	By default, this runs asynchronously, and any frames that were 
	missed while processing are discarded.
	
'''

import urllib.request
            
def get_frames(url, callback, async = True) :
	
	'''
	url: e.g. http://127.0.0.1:8081.
	callback: any function that takes a JPG as a parameter, e.g. callback(jpg)
	async: True by default. Set to false if you are confident of consuming the stream in real time.
	'''
	
	stream = urllib.request.urlopen(url)
	buff = bytearray(0)

	while True :
				  
		# Read frame from URL
		buff += stream.read(1024)
		a = buff.find(b'\xff\xd8') # start code for jpg frame
		b = buff.find(b'\xff\xd9') # end code for jpg frame

		if a != -1 and b != -1:
			jpg = buff[a:b+2]
			buff = buff[b+2:] 
			
			# now invoke the callback
			callback(jpg)
			
			# Flush the stream and reopen, otherwise frames backup
			if async:
				
				buff = bytearray(0)
				stream.close()
				stream = urllib.request.urlopen(url)
