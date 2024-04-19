## Dataset Preparation

For the Siamese model training, 
need organize images into three folders (in data folder):
- **positive**: Contains images for creating positive pairs (images of the same person).
- **negative**: Contains images for creating negative pairs (images of different people).
- **anchor**: Contains anchor images used for comparisons.

### Negative Images

For example, Faces in the Wild (LFW) database.

Download the dataset from the following link:

[Labeled Faces in the Wild (LFW) Database](https://vis-www.cs.umass.edu/lfw/)

**Download all images as a gzipped tar file:**
[All images as gzipped tar file](https://vis-www.cs.umass.edu/lfw/lfw.tgz)

After downloading, extract the files into a temporary folder. To move them into the correct directory, run the `move_negative.py` script:

```bash
python move_negative.py
```
### Anchor and Positive Images

To create anchor and positive images, 
run the cam_create_anchor_and_positive.py which open camera:

```bash
python cam_create_anchor_and_positive.py
```

When pressing and holding 'a' or 'p', 
need slowly rotate head to capture different sides of face

Press and hold 'a' to capture about 400 anchor images.
Press and hold 'p' to capture about 400 positive images.

Images will be saved to 'data/anchor' and 'data/positive' folders.

press 'q' for quit

## Create, train and save model

```bash
python main.py
```

## Test saved model

copy random positive images to application_data/verification_images
copy random positive image to application_data/input_image

```bash
python test_model.py
```

press 'v' for check - script check and print in terminal true or false

press 'q' for quit