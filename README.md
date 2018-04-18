## Install git lfs before cloning

    sudo dnf install git-lfs
    git clone https://github.com/h908714124/hello-hough.git

## Install dependencies

    pip install --user pillow opencv-python tensorflow numpy matplotlib

## Run

    python hough.py --image=test/data/board.png
