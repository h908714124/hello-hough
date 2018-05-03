## Install git lfs before cloning (SuSe: https://github.com/git-lfs/git-lfs/issues/1055)

    sudo dnf install git-lfs
    git clone https://github.com/h908714124/hello-hough.git

## Install python (Fedora)

    sudo dnf install python3-tkinter

## Install python (SuSe)

    sudo dnf install python3-tk python3-pip

## Install dependencies

    pip3 install --user opencv-python tensorflow numpy matplotlib pillow

## Create training data

    python3 hough.py --image=test/data/board.png --create_dataset --train_images=/tmp/train_images --train_labels=/tmp/train_labels

## Read dataset (diagnostic)

    python3 hough.py --read_dataset --train_images=/tmp/train_images

## Show lines (diagnostic)

    python3 hough.py --image=test/data/board.png --show_lines

## Train

    python3 hough.py --train --export_dir=/tmp/trained_go_model --train_images=/tmp/train_images --train_labels=/tmp/train_labels

## Predict

    python3 hough.py --predict --image=test/data/black_center.png --export_dir=/tmp/trained_go_model
