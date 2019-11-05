#!/bin/bash

cat models/msdetector.?? > models/megadetector_v3.pb

celery -A tasks worker --loglevel=info
