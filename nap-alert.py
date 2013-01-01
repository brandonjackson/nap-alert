#!/usr/bin/env python2.7

# Import Libraries
import cv2
import cv2.cv as cv
import numpy
from numpy import array
import time
import math
from collections import deque
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

# Cascade Classifier Wrapper Class
class FaceDetector:

	faceClassifier = cv2.CascadeClassifier(FACE_CLASSIFIER_PATH);
	eyeClassifier = cv2.CascadeClassifier(EYE_CLASSIFIER_PATH);
	
	# Detect eyes and faces, returning two sets of coordinates
	def detect(img, faceRects=False):
		
		# Detect face if rectangle not specified
		if !faceRects:
			faceRects = self.classifyFace(img);
		
		# Ensure 1 face found
		if len(faceRects) is not 1:
			# @todo throw error message
			print "No Face Found!";
			break;

		# Extract face coordinates
		x1,y1,x2,y2 = faceRects[0];
		
		# Extract eyes region of interest (ROI), cropping mouth and hair
		faceHeight = y2-y1;
		y1 = y1 + faceHeight*0.16;
		y2 = y2 - faceHeight*0.32;
		eyesROI = img[y1:y2, x1:x2];

		# Search for eyes
		eyeRects = self.classifyEyes(eyesROI);
		
		# Adjust coordinates to be in faceRects coordinate space
		for e in eyeRects:	
			e[0] += x1;
			e[1] += y1;
			e[2] += x1;
			e[3] += y1;
			
		# @todo split eyes into left and right
		# @todo error checking
		
		return faceRects, eyeRects;

	# Run Cascade Classifier on Image
	def classify(img, cascade, minSizeX=40):
	
		# Run Cascade Classifier
		rects = cascade.detectMultiScale(
				img, minSize=(minSizeX,minSizeX), 
				flags=cv.CV_HAAR_SCALE_IMAGE);
		
		# No Results
		if len(rects) == 0:
			return [];
		
		rects[:,2:] += rects[:,:2]; # ? ? ? 
		return rects;
	
	# Run Face Cascade Classifier on Image
	def classifyFace(img):
		return self.classify(img,self.faceClassifier,FACE_MIN_SIZE);
	
	# Run Eyes Cascade Classifier on Image
	def classifyEyes(img):
		return self.classify(img,self.eyeClassifier,EYE_MIN_SIZE);

class FaceModel:

	QUEUE_MAXLEN = 25;
	
		
	
class EyeHistory:
	
	QUEUE_MAXLEN = 15;
	
	eyeLeft = deque(maxlen=QUEUE_MAXLEN);
	eyeRight = deque(maxlen=QUEUE_MAXLEN);
	eyeLeftMean = [0, 0, 0, 0];
	eyeRightMean = [0, 0, 0, 0];
		
	def add(self,eyeRects):
	
		# Loop over each eye
		for e in eyeRects:
		
			e = array(e); # convert to numpy array
			
			# Distance b/w top left pixel of old/new left/right eye positions
			# @todo find 2D distance using pythagoras' theorem
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
		eyeLeftSum = array([0, 0, 0, 0]);
		for e in self.eyeLeft:
			eyeLeftSum += e;
		if len(self.eyeLeft) > 0:
			self.eyeLeftMean = eyeLeftSum / len(self.eyeLeft);
			
		# Calculate Eye Two Mean
		eyeRightSum = array([0, 0, 0, 0]);
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
	# @todo flush eye history whenever faceRects midpoint changes
	# @todo flush eye history whenever eye rectangle outside of faceRects bbox
	# @todo make sure that eye rectangles don't overlap
	
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
