## Install git lfs before cloning

    sudo dnf install git-lfs
    git clone https://github.com/h908714124/hello-hough.git

## Install tkinter

    sudo dnf install python3-tkinter

## Install dependencies

    pip3 install --user opencv-python tensorflow numpy matplotlib

## Create dataset

    python3 hough.py test/data/board.png --create_dataset --dataset_images=/tmp/images

## Read dataset

    python3 hough.py --read_dataset --dataset_images=/tmp/images
