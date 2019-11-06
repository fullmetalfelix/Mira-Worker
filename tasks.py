import os  
import time  
from celery import Celery
from pymongo import MongoClient
import gridfs
import io
import sys
from datetime import datetime
from enum import IntEnum, Enum, unique
import traceback

import numpy as np
import math
from io import StringIO, BytesIO
import base64

import tensorflow as tf
from PIL import Image


REDIS_PASSWORD = os.environ['REDIS_PASSWORD']
REDIS_HOST = os.environ['REDIS_HOST']
CELERY_BROKER_URL		= 'redis://:{}@{}:6379/0'.format(REDIS_PASSWORD, REDIS_HOST)
CELERY_RESULT_BACKEND	= 'redis://:{}@{}:6379/0'.format(REDIS_PASSWORD, REDIS_HOST)

celery = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery.conf.update(
	enable_utc=True,
	task_serializer='pickle',
	result_serializer='json',
	accept_content=['pickle', 'json'],
	task_track_started=True
)


MONGO_USER = os.environ['MONGO_USER']
MONGO_PWD  = os.environ['MONGO_PWD']
MONGO_HOST = os.environ['MONGO_HOST']
MONGO_DB = os.environ['MONGO_DB']

db = MongoClient(MONGO_HOST,
		username=MONGO_USER,
		password=MONGO_PWD,
		authSource=MONGO_DB,
)[MONGO_DB]
fs = gridfs.GridFS(db)


MAX_ARCH_SIZE = 12582912


@celery.task(bind=True, name='mira.detect')
def task_detect(self, imgID):


	imageinfo = db.images.find_one({'_id': imgID}, {'thumb': 0})

	# do the TF run
	crops = MegaScan(imageinfo)

	if len(crops) != 0:

		# load our scanners -------------------------------------
		mfile = "models/FLP_Foxer_Model1.0-aug.82-0.3369-0.8745"
		json_file = open(mfile + ".json", 'r'); loaded_model_json = json_file.read(); json_file.close()
		modelL = model_from_json(loaded_model_json)
		modelL.load_weights(mfile + ".hdf5")
		modelL = {
			'model': modelL,
			'name': 'Foxer',
			'fullname': 'FLP_Foxer_Model1.0-aug.82-0.3369-0.8745',
			'classes': ['fox', 'skunk', 'empty']
		}



		mfile = "models/model-ratter-20181130"
		json_file = open(mfile + ".json", 'r'); loaded_model_json = json_file.read(); json_file.close()
		modelS = model_from_json(loaded_model_json)
		modelS.load_weights(mfile + ".hdf5")
		modelS = {
			'model': modelS,
			'name': 'ratter',
			'fullname': 'model-ratter-20181130',
			'classes': ['rodent', 'empty']
		}
		# -------------------------------------------------------



		# prepare the image for our scanners --------------------
		dataurl = imageinfo['file']; # print(dataurl[0:25])
		dataurl = dataurl.split(';base64,')[1]
		dataurl = base64.b64decode(dataurl)
		image = Image.open(BytesIO(dataurl))
		isGray = True

		# determine if the image is color or BW
		# this is done by checking pixels in a line in the middle of the image
		# if the RBG channels are the same, then it is grayscale

		for i in range(image.size[0]):
			c = image.getpixel((i,image.size[1] / 2))
			c = np.asarray(c)
			mean = np.mean(c)
			tmp = True
			for b in c: tmp = tmp and (b == mean)
			if not tmp:
				isGray = False
				break
	    
		# colour images will be converted to grayscale for now
		if not isGray: image = image.convert('L')

		W, H = image.size
		# -------------------------------------------------------


		# take each crop and send it to our classifiers
		for cropinfo in crops:

			box = cropinfo['coords']
			boxpx = [int(box[1]*W), int(box[0]*H), int(box[3]*W), int(box[2]*H)]
			cropresults = []

			for model in [modelL, modelS]:

				input_shape = model['model'].get_layer(index=0).input_shape[2:]
				insize = (1,1, input_shape[0], input_shape[1])

				crp = image.crop(boxpx)
				crp = crp.resize(input_shape, Image.LANCZOS)

				# this procedure has to be the same as the one in datagen used for training
				# the pixel data is numerically shifted around its mean and scaled by its stdev
				crp = np.asarray(crp).astype(np.float32)
				crp /= 255
				if len(crp.shape) == 3: crp = crp[:,:,0]

				crp -= np.mean(crp)
				stdv = np.std(crp)
				if stdv == 0: stdv = 1
				crp /= stdv

				cropdata = np.zeros(insize, dtype=np.float32)
				cropdata[0,0,] = crp

				pred = model['model'].predict(cropdata, verbose=0)[0]
				
				output = {}
				for i in range(len(pred)): output[model['classes'][i]] = pred[i]

				analysis = {
					'name': model['name'],
					'fullname': model['fullname'],
					'result': output,
				}
				cropresults.append(analysis)
			
			cropinfo['analysis'] = cropresults


	# code here means the image is still there most likely
	# store crops in the database
	edits = {
		'crops': crops,
		'phase': 10, # detector done
	}
	if len(crops) == 0: edits['phase'] = -10

	db.images.update_one({'_id': imageinfo['_id']},
		{'$set': edits}
	)

	# kill the task associated with this image
	db.tasks.delete_one({'ctask': task_detect.request.id})

	return 0



def generate_detections(detection_graph, images):
	"""
	boxes,scores,classes,images = generate_detections(detection_graph,images)
	Run an already-loaded detector network on a set of images.
	[images] can be a list of numpy arrays or a list of filenames.  Non-list inputs will be
	wrapped into a list.
	Boxes are returned in relative coordinates as (top, left, bottom, right); 
	x,y origin is the upper-left.

	[boxes] will be returned as a numpy array of size nImages x nDetections x 4.

	[scores] and [classes] will each be returned as a numpy array of size nImages x nDetections.

	[images] is a set of numpy arrays corresponding to the input parameter [images], which may have
	have been either arrays or filenames.    
	"""

	if not isinstance(images,list):
		images = [images]
	else:
		images = images.copy()


	boxes = []
	scores = []
	classes = []

	nImages = len(images)

	with detection_graph.as_default():
	    
		with tf.compat.v1.Session(graph=detection_graph) as sess:

			iImage = 0
			imageNP = images[0]

			imageNP_expanded = np.expand_dims(imageNP, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			box = detection_graph.get_tensor_by_name('detection_boxes:0')
			score = detection_graph.get_tensor_by_name('detection_scores:0')
			clss = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			# Actual detection
			(box, score, clss, num_detections) = sess.run(
				[box, score, clss, num_detections],
				feed_dict={image_tensor: imageNP_expanded})

			boxes.append(box)
			scores.append(score)
			classes.append(clss)


		# ...with tf.Session

	# ...with detection_graph.as_default()

	nBoxes = len(boxes)

	# Currently "boxes" is a list of length nImages, where each element is shaped as
	#
	# 1,nDetections,4
	#
	# This implicitly banks on TF giving us back a fixed number of boxes, let's assert on this
	# to make sure this doesn't silently break in the future.
	nDetections = -1
	# iBox = 0; box = boxes[iBox]
	for iBox,box in enumerate(boxes):
		nDetectionsThisBox = box.shape[1]
		assert (nDetections == -1 or nDetectionsThisBox == nDetections), 'Detection count mismatch'
		nDetections = nDetectionsThisBox
		assert(box.shape[0] == 1)

	# "scores" is a length-nImages list of elements with size 1,nDetections
	assert(len(scores) == nImages)
	for(iScore,score) in enumerate(scores):
		assert score.shape[0] == 1
		assert score.shape[1] == nDetections


	# "classes" is a length-nImages list of elements with size 1,nDetections
	#
	# Still as floats, but really representing ints
	assert(len(classes) == nBoxes)
	for(iClass,c) in enumerate(classes):
		assert c.shape[0] == 1
		assert c.shape[1] == nDetections


	# Squeeze out the empty axis
	boxes = np.squeeze(np.array(boxes),axis=1)
	scores = np.squeeze(np.array(scores),axis=1)
	classes = np.squeeze(np.array(classes),axis=1).astype(int)

	# boxes is nImages x nDetections x 4
	assert(len(boxes.shape) == 3)
	assert(boxes.shape[0] == nImages)
	assert(boxes.shape[1] == nDetections)
	assert(boxes.shape[2] == 4)

	# scores and classes are both nImages x nDetections
	assert(len(scores.shape) == 2)
	assert(scores.shape[0] == nImages)
	assert(scores.shape[1] == nDetections)

	assert(len(classes.shape) == 2)
	assert(classes.shape[0] == nImages)
	assert(classes.shape[1] == nDetections)

	return boxes,scores,classes,images



def MegaScan(image):

	dataurl = image['file']; # print(dataurl[0:25])
	dataurl = dataurl.split(';base64,')[1]
	dataurl = base64.b64decode(dataurl)

	img = Image.open(BytesIO(dataurl))

	# prepare the image for the megascanner
	nChannels = 3
	imgMS = np.array(img)
	if len(imgMS.shape) == 2:
		# simple 1channel image
		tmp = np.zeros((imgMS.shape[0], imgMS.shape[1], 3));
		tmp[:,:,0] = imgMS
		tmp[:,:,1] = imgMS
		tmp[:,:,2] = imgMS
		imgMS = tmp

	else:

		nChannels = imgMS.shape[2]
		if nChannels > 3:
			print('Warning: trimming channels from image')
			imgMS = imgMS[:,:,0:3]

	

	print('loading TF...')
	import tensorflow as tf
	
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	# LOADS THE MS MEGADETECTOR
	DETECTOR = tf.compat.v1.Graph()
	with DETECTOR.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		with tf.io.gfile.GFile('./models/megadetector_v3.pb', 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	print('detector loaded!')


	boxes,scores,classes,images = generate_detections(DETECTOR, imgMS)
	boxes = boxes[0]
	scores = scores[0]
	classes = classes[0]
	ts = datetime.utcnow()

	MEGA_CONFIDENCE_THRESHOLD = 0.85

	# filter the sure boxes
	crops = []
	for i in range(len(boxes)):

		#print(boxes[i], scores[i], classes[i])

		if scores[i] < MEGA_CONFIDENCE_THRESHOLD: continue

		crop = {
			'coords': boxes[i].tolist(),
			'score': float(scores[i]),
			'class': int(classes[i]),
			'animal': 'unknown',
			'time': ts,
			'detector': 'MEGADETECTOR',
		}
		crops.append(crop)

	return crops




