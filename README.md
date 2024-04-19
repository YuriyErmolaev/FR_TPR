## Dataset Preparation

For the Siamese model training, organize images into three distinct folders:
- **Positive**: Contains images for creating positive pairs (images of the same person).
- **Negative**: Contains images for creating negative pairs (images of different people).
- **Anchor**: Contains anchor images used for comparisons.

### Negative Images

For example, Faces in the Wild (LFW) database.

Download the dataset from the following link:

[Labeled Faces in the Wild (LFW) Database](https://vis-www.cs.umass.edu/lfw/)

**Download all images as a gzipped tar file:**
[All images as gzipped tar file](https://vis-www.cs.umass.edu/lfw/lfw.tgz)

After downloading, extract the files into a temporary folder. To move them into the correct directory, run the `move_negative.py` script:

```bash
python move_negative.py
