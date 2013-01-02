#!/usr/bin/env python2.7

"""
nap-alert
by Brandon Jackson

nap-alert.py
Main python script
"""

# Import Libraries
import time
import math
from collections import deque

import numpy
import cv2
import cv2.cv as cv
import wx
 
# Constants
CAMERA_INDEX = 0;
SCALE_FACTOR = 3; # video size will be 1/SCALE_FACTOR
FACE_CLASSIFIER_PATH = "classifiers/haar-face.xml";
EYE_CLASSIFIER_PATH = "classifiers/haar-eyes.xml";
FACE_MIN_SIZE = 80;
EYE_MIN_SIZE = 10;

# Setup Webcam
capture = cv2.VideoCapture(CAMERA_INDEX);

# Reduce Video Size to make Processing Faster
height = capture.get(cv.CV_CAP_PROP_FRAME_WIDTH)/SCALE_FACTOR;
width = capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT)/SCALE_FACTOR;
capture.set(cv.CV_CAP_PROP_FRAME_WIDTH,height);
capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT,width);

# Create window
cv2.namedWindow("Video", cv2.CV_WINDOW_AUTOSIZE);

class FaceDetector:

	"""
	FaceDetector is a wrapper for the cascade classifiers.
	Should be initialized once per program instance.
	"""

	# Load Haar Cascade Classifiers
	faceClassifier = cv2.CascadeClassifier(FACE_CLASSIFIER_PATH);
	eyeClassifier = cv2.CascadeClassifier(EYE_CLASSIFIER_PATH);
	
	# 
	# Args
	def detect(self,img, faceRects=False):
		"""
		Detect face and eyes. 
		Runs Haar cascade classifiers. Sometimes it is desirable to speed up 
		processing by using a previously-found face rectangle. To do this, pass 
		the old faceRects as the second argument.
		
		Args:
			img (numpy array): input image
			faceRects (numpy array): array of face rectangle. Face detected if 
									 omitted.
		Returns:
			a dictionary with three elements each representing a rectangle
		"""

		# Data structure to hold frame info
		rects = {
			'face': [],
			'eyeLeft': [],
			'eyeRight': []
		};
		
		# Detect face if old faceRects not provided
		if not faceRects:
			faceRects = self._classifyFace(img);
		
		# Ensure a single face found
		if len(faceRects) > 1:
			# TODO throw error message
			print "Multiple Faces Found!";
			return;
		if len(faceRects) == 0:
			# TODO throw error message, or perhaps not in this case
			return rects;

		rects['face'] = faceRects[0];

		# Extract face coordinates, calculate center and diameter
		x1,y1,x2,y2 = rects['face'];
		faceCenter = (((x1+x2)/2.0), ((y1+y2)/2.0));
		faceDiameter = y2-y1;
		
		# Extract eyes region of interest (ROI), cropping mouth and hair
		eyesY1 = y1 + (faceDiameter * 0.16);
		eyesY2 = y2 - (faceDiameter * 0.32);
		eyesROI = img[eyesY1:eyesY2, x1:x2];

		# Search for eyes in ROI
		eyeRects = self._classifyEyes(eyesROI);
		
		# Ensure (at most) two eyes found
		if len(eyeRects) > 2:
			# TODO throw error message
			print "Multiple Eyes Found!";
			return;

		# Loop over each eye
		for e in eyeRects:
			# Adjust coordinates to be in faceRects coordinate space
			e += rects['face'];
			
			# Split left and right eyes. Compare eye and face midpoints.
			eyeMidpointX = (e[0]+e[2])/2.0;
			if eyeMidpointX < faceCenter[0]:
				rects['eyeLeft'] = e; # TODO prevent overwriting
			else:
				rects['eyeRight'] = e;
		# TODO error checking
		# TODO calculate signal quality
		
		return rects;

	# Run Cascade Classifier on Image
	def _classify(self,img, cascade, minSizeX=40):
	
		# Run Cascade Classifier
		rects = cascade.detectMultiScale(
				img, minSize=(minSizeX,minSizeX), 
				flags=cv.CV_HAAR_SCALE_IMAGE);
		
		# No Results
		if len(rects) == 0:
			return [];
		
		rects[:,2:] += rects[:,:2]; # ? ? ? 
		rects = numpy.array(rects);
		return rects;
	
	# Run Face Cascade Classifier on Image
	def _classifyFace(self,img):
		return self._classify(img,self.faceClassifier,FACE_MIN_SIZE);
	
	# Run Eyes Cascade Classifier on Image
	def _classifyEyes(self,img):
		return self._classify(img,self.eyeClassifier,EYE_MIN_SIZE);

class FaceModel:

	"""
	FaceModel integrates data from the new frame into a model that keeps track of where the eyes are. To do this it uses:
		- A moving average of the most recent frames
		- Facial geometry to fill in missing data
	The resulting model generates a set of two specific regions of interest (ROI's) where blinking is expected to take place.
	"""

	QUEUE_MAXLEN = 25;
	
	# Queues storing most recent position rectangles, used to calculate
	# moving averages
	rectHistory = {
		'face': deque(maxlen=QUEUE_MAXLEN),
		'eyeLeft': deque(maxlen=QUEUE_MAXLEN),
		'eyeRight': deque(maxlen=QUEUE_MAXLEN)
	};
	
	# Moving average of position rectangles
	rectAverage = {
		'face': [],
		'eyeLeft': [],
		'eyeRight': []
	};
	
	def add(self,rects):
		"""Add new set of rectangles to model"""
		
		# Checks to see if face has moved significantly. If so, resets history.
		if(self._faceHasMoved(rects['face'])):
			self.clear();
		
		# Loop over face, eyeLeft and eyeRight, adding rects to history
		for key,rect in rects.items():
			# Skip empty members
			if rect is []:
				continue;
			# Add to position average history
			self.rectHistory[key].append(rect);
		
		# Update moving average stats
		self._updateAverages();
	
	def getEyeRects(self):
		"""Get array of eye rectangles"""
		return [self.rectAverage['eyeLeft'], self.rectAverage['eyeRight']];

	def getEyeLine(self):
		"""Returns Points to create line along axis of eyes"""
		left,right = self.getEyeRects();
		leftPoint = (left[0], ((left[1] + left[3])/2));
		rightPoint = (right[2], ((right[1] + right[3])/2));
		return [leftPoint,rightPoint];
		
	def clear(self):
		""" Resets Eye History"""
		self.rectAverage['eyeLeft']=[];
		self.rectAverage['eyeRight']=[];
		self.rectHistory['eyeLeft'].clear();
		self.rectHistory['eyeRight'].clear();

	def _faceHasMoved(self, recentFaceRect):
		"""Determines if face has just moved, requiring history reset"""
	
		# If no face found, return true
		if(len(recentFaceRect) is 0):
			return True;

		history = self.rectHistory['face'];
		
		if len(history) is not self.QUEUE_MAXLEN:
			return False;

		old = history[self.QUEUE_MAXLEN-3];
		oldX = (old[0] + old[2]) / 2.0;
		oldY = (old[1] + old[3]) / 2.0;
		recentX = (recentFaceRect[0] + recentFaceRect[2]) / 2.0;
		recentY = (recentFaceRect[1] + recentFaceRect[3]) / 2.0;
		change = ((recentX-oldX)**2 + (recentY-oldY)**2)**0.5; # magnitude i.e. sqrt(a^2+b^2)
		return True if change > 20 else False;

	def _updateAverages(self):
		"""Update position rectangle moving averages"""
		for key,queue in self.rectHistory.items():
			self.rectAverages[key] = sum(queue) / float(len(queue));
	
class EyeHistory:
	
	QUEUE_MAXLEN = 15;
	
	eyeLeft = deque(maxlen=QUEUE_MAXLEN);
	eyeRight = deque(maxlen=QUEUE_MAXLEN);
	eyeLeftMean = [0, 0, 0, 0];
	eyeRightMean = [0, 0, 0, 0];
		
	def add(self,eyeRects):
	
		# Loop over each eye
		for e in eyeRects:
		
			e = numpy.array(e); # convert to numpy array
			
			# Distance b/w top left pixel of old/new left/right eye positions
			# TODO find 2D distance using pythagoras' theorem
			diffLeft = abs(e[0] - self.eyeLeftMean[0]); 	
			diffRight = abs(e[0] - self.eyeRightMean[0]);
			
			# Eye Closer to Left than Right
			if diffLeft <= diffRight:
				q = self.eyeLeft;
				
				# If right eye empty, encourage splitting 
				# Add right point to eyeRight min
				if len(self.eyeRight) is 0: 
					self.eyeRightMean[0] = e[3];
			else:
				q = self.eyeRight;
				#if (e[0] > self.eyeLeftMean[0] or e[0] < self.eyeLeftMean[2]) or (e[2] > self.eyeLeftMean[0] or e[2] < self.eyeLeftMean[2]):
				#	continue;
				   
			q.append(e);
			self.updateMeans();

		
	def updateMeans(self):
	
		# Calculate Eye One Mean
		eyeLeftSum = numpy.array([0, 0, 0, 0]);
		for e in self.eyeLeft:
			eyeLeftSum += e;
		if len(self.eyeLeft) > 0:
			self.eyeLeftMean = eyeLeftSum / len(self.eyeLeft);
			
		# Calculate Eye Two Mean
		eyeRightSum = numpy.array([0, 0, 0, 0]);
		for e in self.eyeRight:
			eyeRightSum += e;
		if len(self.eyeRight) > 0:
			self.eyeRightMean = eyeRightSum / len(self.eyeRight);

	def getEyeRects(self):
		return [self.eyeLeftMean, self.eyeRightMean];
	
	# Returns Points to create line along axis of eyes
	def getEyeLine(self):
		if(self.eyeLeftMean[0] < self.eyeRightMean[0]):
			left = self.eyeLeftMean;
			right = self.eyeRightMean;
		else:
			left = self.eyeRightMean;
			right = self.eyeLeftMean;
		
		leftPoint = (left[0], ((left[1] + left[3])/2));
		rightPoint = (right[2], ((right[1] + right[3])/2));
		return [leftPoint,rightPoint];

	# Resets history
	def clear(self):
		self.eyeLeftMean = [0,0,0,0];
		self.eyeRightMean = [0,0,0,0];
		self.eyeLeft.clear();
		self.eyeRight.clear();

class FaceHistory:

	QUEUE_MAXLEN = 10;

	faces = deque(maxlen=QUEUE_MAXLEN);
	
	def add(self,faceRect):
		if len(faceRect) is not 1:
			return;
		self.faces.append(faceRect[0]);
	
	def hasMoved(self):
		if len(self.faces) is not self.QUEUE_MAXLEN:
			return False;

		old = self.faces[self.QUEUE_MAXLEN-3];
		recent = self.faces[self.QUEUE_MAXLEN-1];
		oldX = (old[0]+old[2])/2.0;
		oldY = (old[1]+old[3])/2.0;
		recentX = (recent[0]+recent[2])/2.0;
		recentY = (recent[1]+recent[3])/2.0;
		change = ((recentX-oldX)**2 + (recentY-oldY)**2)**0.5; # magnitude i.e. sqrt(a^2+b^2)
		return True if change > 20 else False;		

eyeH = EyeHistory();
faceH = FaceHistory();

# Draw rectangles on image
def drawRects(img, rects, color):
	for x1, y1, x2, y2 in rects:
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

oldTime = time.time();
i = 0;
global faceRects;
while True:
	# Calculate time difference (dt), update oldTime variable
	newTime = time.time();
	dt =  newTime - oldTime;
	oldTime = newTime;
	
	# Grab Frame
	retVal, frame = capture.read();
	displayFrame = frame.copy();
	gray = cv2.equalizeHist(cv2.cvtColor(frame,cv.CV_BGR2GRAY));
	
	# Detect Face 20% of the Time
	if i % 5 is 0:
		faceRects = detect(gray,faceDetector,80);
		faceH.add(faceRects);
		if faceH.hasMoved():
			eyeH.clear();
			print 'flushing eyes.';
	i += 1;

	# Ignore detection results if no/multiple face(s) found
	if len(faceRects) is not 1:
		print "No Face Found!"
		continue;
	
	# Extract face coordinates
	x1,y1,x2,y2 = faceRects[0];
	
	# Crop out mouth and hair 
	faceHeight = y2-y1;
	y1 = y1 + faceHeight*0.16;
	y2 = y2 - faceHeight*0.32;

	# Extract face region of interest (ROI)
	faceROI = gray[y1:y2, x1:x2];
	# faceROI = cv2.equalizeHist(faceROI); # equalizes eye region of interest
		
	# Detect Eyes
	eyeRects = detect(faceROI,eyeDetector,10);
	print eyeRects;
	for c in eyeRects:	# Adjust coordinates to be in faceRects coordinate space
		c[0] += x1;
		c[1] += y1;
		c[2] += x1;
		c[3] += y1;
	
	# Get mean eyeRects coordinates
	eyeH.add(eyeRects);
	eyeRects = eyeH.getEyeRects();
	# TODO flush eye history whenever faceRects midpoint changes
	# TODO flush eye history whenever eye rectangle outside of faceRects bbox
	# TODO make sure that eye rectangles don't overlap
	
	linePoints = eyeH.getEyeLine();
	cv2.line(displayFrame, linePoints[0],linePoints[1],(0, 0, 255));
	
	drawRects(displayFrame, faceRects, (0, 0, 255));
	drawRects(displayFrame, eyeRects, (0, 255, 0));

	cv2.imshow("Video", displayFrame);
	
	# Show face ROI
	faceDisplayFrame = faceROI.copy();
	cv2.imshow('Display face ROI', faceDisplayFrame);

	# If no eyes found, skip next section
	if len(eyeRects[0]) is 0 or len(eyeRects[1]) is 0 or eyeRects[1][2] is 0:
		continue;
	
	eyeLeftROI = frame[eyeRects[0][1]:eyeRects[0][3],eyeRects[0][0]:eyeRects[0][2]];
	eyeRightROI = frame[eyeRects[1][1]:eyeRects[1][3],eyeRects[1][0]:eyeRects[1][2]];
	#eyeLeftROI = cv2.equalizeHist(eyeLeftROI);
	#eyeRightROI = cv2.equalizeHist(eyeRightROI);
	eyeLeftVis = eyeLeftROI.copy();
	eyeRightVis = eyeRightROI.copy();
	cv2.imshow("eyeLeft",eyeLeftVis);
	cv2.moveWindow("eyeLeft",300,200);
	cv2.imshow("eyeRight",eyeRightVis);
	cv2.moveWindow("eyeRight",300,260);
