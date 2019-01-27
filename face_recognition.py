import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# capture frames from a camera
cap = cv.VideoCapture(0)

# loop runs if capturing has been initialized.
while 1:

	# reads frames from a camera
	ret, img = cap.read()

	# convert to gray scale of each frames
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# Detects faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		# To draw a rectangle in a face
		cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		# Detects eyes of different sizes in the input image
		eyes = eye_cascade.detectMultiScale(roi_gray)

		#To draw a rectangle in eyes
		for (ex,ey,ew,eh) in eyes:
			cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

	# Display an image in a window
	cv.imshow('img',img)

	# Wait for Esc key to stop
	k = cv.waitKey(30) & 0xff
	if k == 27:
		break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv.destroyAllWindows()
