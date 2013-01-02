nap-alert
=========

By Brandon Jackson
Project started December 2012

*nap-alert* is a fatigue detection system that alerts users when they are falling asleep. The app is designed to combat [narcolepsy][1] and [microsleep][2].

*nap-alert* was built using the [OpenCV][3] framework.

How It Works
------------

Microsleep strikes quickly. You probably don't even realize you're in the process of falling asleep, and almost certainly don't notice that you are blinking for longer than usual. *nap-alert* uses your computer's webcam to monitor your blink rate and average blink duration. If your eyes stay closed for long periods of time, then the program lets you know that a nap is imminent. Research indicates that this is one of the most effective ways to predict impending microsleep. For a good review of recent experimental findings, see [Caffier et al][4].

System Requirements
-------------------

*nap-alert* should work on any OSX, Linux or Windows system with an OpenCV-compatible webcam. OpenCV maintains a list of these cameras [here][5].

Code Overview
-------------

1. Initialization
	- Webcam initialized, capture size scaled down for speed
	- `FaceModel` and `BlinkStats` classes are initialized. These classes synthesize information from multiple frames into a more complete picture.
2. *Controller* Loop:
	- The highest-level `while` loop which initiates each frame of image capture
	- On each new frame the following information is gathered and sent to the `BlinkStats` class (see below):
		1. runs the *image processing loop* (see below) which returns a guess as to whether the image captures a blink event and a reliability indicator
		2. measures the time that has elapsed since the last frame
3. The Image Processing Loop: each new frame from the webcam goes through the following steps of processing.
	1. `FaceDetector` scans the image for faces and eyes, and returns a set of rectangles using Haar Cascade Classifiers.
	2. `FaceModel` integrates data from the new frame into a model that keeps track of where the eyes are. To do this it uses:
		- A moving average of the most recent frames
		- Facial geometry to fill in missing data
	The resulting model generates a set of two specific regions of interest (ROI's) where blinking is expected to take place.
	3. `BlinkDetector` watches the blinking ROI for blinking. Detection uses a variety of techniques:
		- If eye rectangles suddenly disappear and then quickly reappear, this is interpreted as a blink. This method only used when signal quality is very high, to prevent false positives.
		- @todo Watches for increase increased brightness in red channel of ROI, caused by the presence of more red pigment in the eyelid than in the eye itself.
		- @todo Tracks rapid motion in the ROI
	The class returns a guess as to whether the current frame captures a blink event, along with a signal quality / confidence / reliability indicator
4. `BlinkStats` receives blink, time and reliability information and does high-level analysis:
	- keeps track of blink event durations and inter-blink intervals
	- determines how much weight to give to blink event based on reliability metric
	- finds larger-scale blink trends, such as a slow ramping up of blink duration that corresponds to increasing sleepiness
5. If `BlinkStats` finds dangerous trends, the app broadcasts a *NAP ALERT!*

Notes
-----

- Designed and tested using the iSight camera of a 15" Late 2011 Macbook Pro

[1]: http://en.wikipedia.org/wiki/Narcolepsy "Wikipedia: Narcolepsy"
[2]: http://en.wikipedia.org/wiki/Microsleep "Wikipedia: Microsleep"
[3]: http://www.opencv.org "OpenCV Framework Website"
[4]: http://link.springer.com/article/10.1007%2Fs00421-003-0807-5 "Caffier, Philipp P., Udo Erdmann, and Peter Ullsperger. \"Experimental evaluation of eye-blink parameters as a drowsiness measure.\" European Journal of Applied Physiology 89.3 (2003): 319-325."
[5]: http://opencv.willowgarage.com/wiki/Welcome/OS "List of webcams compatible with OpenCV"