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

import numpy
import math
from io import StringIO, BytesIO
import base64



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

db = MongoClient('mongodb://{}:{}@{}'.format(MONGO_USER, MONGO_PWD, MONGO_HOST))[MONGO_DB]
fs = gridfs.GridFS(db)


MAX_ARCH_SIZE = 12582912




@celery.task(bind=True, name='mira.largefilestorage')
def task_largefilestore(self, fulldata, metadata):

	file = fs.new_file(chunkSize=MAX_ARCH_SIZE, metadata=metadata) # 8 MiBytes chunk size
	file.write(fulldata)
	file.close()

	db.tasks.find_one_and_update({'ctask': task_largefilestore.request.id},
		{'$set': {'fileID': file._id}}
	)

	return 0



@celery.task(bind=True, name='mira.detect')
def task_detect(self, userinfo, imageinfo):







	
	return 0








