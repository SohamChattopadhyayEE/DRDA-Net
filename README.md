# DRDA-Net
Dense Residual Dual-shuffle Attention Net (DRDA-Net) : an efficient framework for breast cancer detection. Overall workflow of the proposed framework is shown below- ![flow diagram](https://github.com/SohamChattopadhyayEE/DRDA-Net/blob/main/figures/Overall%20flow%20diagram.jpg)

## The Model
- The entire DRDA-Net model in comprised of several building blocks. The most elementary building block is the `Dual-shuffle Residual Block` or the `DRB`. 
- In addition to that, there are the `Channel Attention (CA)` and the `Residual Dual-shuffle Block (RDAB)`. 
- The illustrative diagram of the proposed DRDA-Net is shown below-
![model](https://github.com/SohamChattopadhyayEE/DRDA-Net/blob/main/figures/Model.JPG)

## Dependencies 
    pip install -r requirements.txt

## Arguments
    D:\directory> python train.py --help
    usage: main.py [-h] [-t_path TRAIN_PATH] [-v_path VALIDATION_PATH] [-p_path PLOT_PATH] [-m_path MODEL_PATH] [-bs BS] [-n N_CLASS] [-lr LR] [-e EPOCH]
                   [-n_e_block NUM_ELEMENTARY_BLOCKS]

    DRDA-Net

    optional arguments:
      -h, --help            show this help message and exit
      -t_path TRAIN_PATH, --train_path TRAIN_PATH
                            path to the files of training images
      -v_path VALIDATION_PATH, --validation_path VALIDATION_PATH
                            path to the files of validation images
      -p_path PLOT_PATH, --plot_path PLOT_PATH
                            path to the convergence plots
      -m_path MODEL_PATH, --model_path MODEL_PATH
                            path to the model.pt
      -bs BS, --bs BS       batch size
      -n N_CLASS, --n_class N_CLASS
                            number of classes
      -lr LR, --lr LR       number of classes
      -e EPOCH, --epoch EPOCH
                            number of epochs
      -n_e_block NUM_ELEMENTARY_BLOCKS, --num_elementary_blocks NUM_ELEMENTARY_BLOCKS
                            number of elementary blocks
## Code Execution
    D:\directory> python app.py
