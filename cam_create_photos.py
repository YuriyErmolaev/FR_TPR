import cv2
import os
import uuid
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print('gpus: ', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print('cur gpu: ', gpu)

POS_PATH = os.path.join('data', 'positive')
ANC_PATH = os.path.join('data', 'anchor')
NEG_PATH = os.path.join('data', 'negative')


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Cut down frame to 250x250px
    frame = frame[120:120 + 250, 200:200 + 250, :]

    key = cv2.waitKey(1)

    if key != -1:  # Check if a key was pressed
        # Collect anchors
        # if cv2.waitKey(1) & 0XFF == ord('a'):
        if key & 0xFF == ord('a'):
            # Create the unique file path
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)

        # Collect positives
        # if cv2.waitKey(1) & 0XFF == ord('p'):
        if key & 0xFF == ord('p'):
            # Create the unique file path
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out positive image
            cv2.imwrite(imgname, frame)

        # Collect negatives
        if key & 0xFF == ord('n'):
            # Create the unique file path
            imgname = os.path.join(NEG_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out negative image
            cv2.imwrite(imgname, frame)

        # Breaking gracefully
        # if cv2.waitKey(1) & 0XFF == ord('q'):
        if key & 0XFF == ord('q'):
            break

    # Show image back to screen
    cv2.imshow('Image Collection', frame)



# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()