## Install git lfs before cloning

    sudo dnf install git-lfs
    git clone https://github.com/h908714124/hello-hough.git

## Install tkinter

    sudo dnf install python3-tkinter

## Install dependencies

    pip3 install --user opencv-python tensorflow numpy matplotlib

## Create training data

    python3 hough.py --image=test/data/board.png --create_dataset --train_images=/tmp/train_images --train_labels=/tmp/train_labels

## Read dataset (diagnostic)

    python3 hough.py --read_dataset --dataset_images=/tmp/images

## Show lines (diagnostic)

    python3 hough.py --image=test/data/board.png --show_lines

## Train

    python3 hough.py --train --export_dir=/tmp/trained_go_model --train_images=/tmp/train_images --train_labels=/tmp/train_labels

## Predict

    python3 hough.py --predict --image=test/data/black_center.png
