__author__ = "Rubén Buzón Pérez"
__email__ = "ruben.buzon@live.u-tad.com"


from turtle import left
from xmlrpc.client import Boolean
import tensorflow as tf
import numpy as np
import cv2
import time
from mss import mss
from PIL import Image
from time import gmtime, strftime


img_height = 224
img_width = 224


class Face:

	FRAMES_TO_UPDATE = 5

	def __init__(self, id, x1, y1, x2, y2, lst, lsf) -> None:
		"""Tracks face between frames using distances between multiple frames

		Parameters
		----------
		id : int
			unique identifier
		x1 : int
			x1 of face
		y1 : int
			y1 of face
		x2 : int
			x2 of face
		y2 : int
			y1 of face
		lst : float
			last seen timestamp
		lsf : int
			last seen frame
		"""

		self.id = id

		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2

		self.lst = lst
		self.lsf = lsf

		self.emotion = 0
		self.frames_emotion = []

	def register_emotion(self, emotion):
		"""Registers an emotion in last frame
		Parameters
		----------
		emotion : int
			emotion detected

		Returns
		-------
		bool
			True if lasf FRAMES_TO_UPDATE frames a different emotion than the last sure emotion was detected. Otherwise, returns False.
		"""
		self.frames_emotion.append(emotion)

		if len(self.frames_emotion) > self.FRAMES_TO_UPDATE:
			self.frames_emotion.pop(0)

		if len(list(set(self.frames_emotion))) == 1 and self.frames_emotion[0] != self.emotion:
			self.emotion = emotion
			return True
		else:
			return False



class EmotionDetector:
	def __init__(self, model, classifier, classes, background ) -> None:
		"""Initialized the object with the attributes passed by parameters

		Parameters
		----------
		model : str
			Location of file containing keras model for detecting facial emotions
		classifier : str
			Location of file containing haar cascade model for detecting faces
		classes : list[str]
			List containing the name of classes that the model detects
		background : bool
			Execute in background, no window will be shown

		"""
		self.model = tf.keras.models.load_model(model)
		self.classifier = cv2.CascadeClassifier(classifier)
		self.classes = classes
		self.background = background

		self.video = cv2.VideoCapture(0)
		self.resX = 1024
		self.resY = 720
		self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.resX)
		self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resY)

		self.frame_counter = 0
		self.faces_counter = 0
		self.faces = []

	def track_face(self, x1, y1, x2, y2) -> int:
		# TODO: evitar duplicados
		"""Tracks face between frames using distances between multiple frames

		Parameters
		----------
		x1 : int
			x1 of face
		y1 : int
			y1 of face
		x2 : int
			x2 of face
		y2 : int
			y1 of face

		Returns
		-------
		Face
			Face object of detected face, will be the same across different frames
		"""
		cur_face = None
		lf_intersection = 0

		for _f in self.faces:
			# Calculate intersection of face with registered faces 
			dx = min(x2, _f.x2) - max(x1, _f.x1)
			dy = min(y2, _f.y2) - max(y1, _f.y1)
			if (dx>=0) and (dy>=0) and dx*dy > lf_intersection:
				lf_intersection = dx*dy
				cur_face = _f

		if cur_face == None:
			cur_face = Face(self.faces_counter, x1, y1, x2, y2, time.time(), self.frame_counter)
			self.faces.append(cur_face)
			self.faces_counter += 1
		else:
			cur_face.lst = time.time()
			cur_face.lsf = self.frame_counter

		return cur_face

	def post_frame_processing(self):
		"""Proccessing made after each frame
		"""
		cur_ts = time.time()

		# Removes old faces not seen in 2 seconds or 120 frames (what happens first)
		self.faces = [_f for _f in self.faces if cur_ts - _f.lst < 2 and self.frame_counter - _f.lsf < 120]

		self.frame_counter += 1

	def _detect_frame(self, frame) -> bool:
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		detected = self.classifier.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
		
		for x, y, w, h in detected:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
			cv2.rectangle(frame, (x-1, y), (x+w//2, y-20), (255, 0, 0), -1)
			face = frame[y+5:y+h-5, x+20:x+w-20]
			face = cv2.resize(face, (img_height, img_width)) 
			face_obj = self.track_face(x, y, x+w, y+h)
			
			prediction = self.model.predict(np.array([face.reshape((img_height, img_width, 3))]))
			predicted_class = prediction.argmax()
			predicted_emotion = self.classes[predicted_class]
			predicted_accuracy = np.amax(prediction)

			if face_obj.register_emotion(predicted_class) == True:
				# TODO: implementar narrador?
				print(f"El rostro {face_obj.id + 1} ha cambiado a {predicted_emotion}")

			cv2.putText(frame,f"#{face_obj.id + 1} {predicted_emotion}",(x+10,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
			
		if(not self.background):
			cv2.imshow('Detector de expresiones faciales', frame)
			if cv2.waitKey(5) != -1:
				return False

		self.post_frame_processing()
		return True


	def start_webcam_detection(self) -> None:
		"""Starts an infinite loop for detecting faces using Webcam mode
		"""

		while True:
			_, frame = self.video.read()

			if not self._detect_frame(frame):
				break

		self.video.release()
		cv2.destroyAllWindows()

	def start_screen_detection(self) -> None:
		"""Starts an infinite loop for detecting faces using Screen mode
		"""

		with mss() as sct:
			monitor = sct.monitors[1]
			while True:
				screenShot = sct.grab(monitor)
				img = Image.frombytes(
					'RGB', 
					(screenShot.width, screenShot.height), 
					screenShot.rgb, 
				)
				
				frame = np.array(img)
				frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

				if not self._detect_frame(frame):
					break

		cv2.destroyAllWindows()
