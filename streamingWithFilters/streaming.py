# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

import argparse
import datetime
import threading
import time

import cv2
import imutils
#from adafruit_servokit import ServoKit
from flask import Flask, Response, redirect, render_template, url_for,session
# import the necessary packages
from flask.globals import request
from flask.wrappers import Request
from imutils.video import VideoStream
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect

from FrameManipulation import frameFilter

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
filter = 'none'
move = "stop"
lr = 90
ud = 90
#kit = ServoKit(channels=16)
#kit.servo[0].set_pulse_width_range(600,2610)
#kit.servo[1].set_pulse_width_range(600,2610)

lock = threading.Lock()
async_mode = None
thread = None
# initialize a flask object
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)


def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count})


@app.route("/")
def index():
	global move
	# return the rendered template
	return render_template("index.html",filt=filter, motion=move)

def move_camera():
	# grab global references to the video stream, output frame, and
	# lock variables
	global   move,lr,ud

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it

		time.sleep(.1)
		if(move == "right"):
			lr +=5
			if lr >135:
				lr = 45
			#kit.servo[1].angle = lr
		elif(move == "left"):
			lr -=5
			if lr < 45:
				lr=135
			#kit.servo[1].angle = lr
		elif move == "up":
			ud +=5
			if ud > 135:
				ud=45
			#kit.servo[0].angle = ud
		elif move == "down":
			if ud < 45:
				ud=135
			ud -=5
			#kit.servo[0].angle = ud
		#print("ud: " +str(ud) +" lr: "+str(lr))



def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock







	# loop over frames from the output stream
	while True:

			# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = cv2.flip(frame,1)
		frame = frameFilter(filter,frame)
		time.sleep(.1)

		# grab the current timestamp and draw it on the frame
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
		"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

		# acquire the lock, set the output frame, and release the
		# lock
		with lock:

			outputFrame = frame.copy()


		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/", methods=["POST"])
def filterUpdate():
	global filter,move
	if request.method == 'POST':
		filter = request.form['options']
	return  redirect(url_for('index'))

@app.route("/move", methods=["POST"])
def moveUpdate():
	global filter,move
	if request.method == 'POST':
		move = request.form['motion']
	return  redirect(url_for('index'))



@socketio.event
def my_ping():
    emit('my_pong')


@socketio.event
def connect():
    global thread
    with lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})


@socketio.event
def my_event(message):
	global filter
	session['receive_count'] = session.get('receive_count', 0) + 1
	emit('my_response',
	     {'data': message['data'], 'count': session['receive_count']})
	filter= message['data']

@socketio.on('move_up')
def move_up(message):
	global move
	session['receive_count'] = session.get('receive_count', 0) + 1
	emit('my_move',
	     {'data': message['data'], 'count': session['receive_count']})
	move= message['data']

@socketio.on('move_down')
def move_down(message):
	global move
	session['receive_count'] = session.get('receive_count', 0) + 1
	emit('my_move',
	     {'data': message['data'], 'count': session['receive_count']})
	move= message['data']

@socketio.on('move_left')
def move_left(message):
	global move
	session['receive_count'] = session.get('receive_count', 0) + 1
	emit('my_move',
	     {'data': message['data'], 'count': session['receive_count']})
	move= message['data']

@socketio.on('move_right')
def move_right(message):
	global move
	session['receive_count'] = session.get('receive_count', 0) + 1
	emit('my_move',
	     {'data': message['data'], 'count': session['receive_count']})
	move= message['data']

@socketio.on('move_stop')
def move_stop(message):
	global move
	session['receive_count'] = session.get('receive_count', 0) + 1
	emit('my_move',
	     {'data': message['data'], 'count': session['receive_count']})
	move= message['data']



# check to see if this is the main thread of execution
if __name__ == '__main__':

	t = threading.Thread(target=move_camera, args=(
		))#args["frame_count"],))
	t.daemon = True
	t.start()


	# start the flask app
	#app.run(host=args["ip"], port=args["port"], debug=True,
	#	threaded=True, use_reloader=False)

	socketio.run(app)

# release the video stream pointer
vs.stop()
