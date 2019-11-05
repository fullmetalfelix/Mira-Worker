#!/bin/bash

OUTPUT="$(celery inspect ping -A tasks -d celery@$HOSTNAME)"
if [[ $OUTPUT == *"Error"* ]]; then
	exit 1
fi
exit 0


