## Install git lfs before cloning

    sudo dnf install git-lfs
    git clone https://github.com/h908714124/hello-hough.git

## Install dependencies

    pip install --user pillow opencv-python tensorflow numpy matplotlib

## Create dataset

    python hough.py test/data/board.png --create_dataset --dataset_images=/tmp/images

## Read dataset

    python hough.py --read_dataset --dataset_images=/tmp/images
