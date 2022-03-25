# DRDA-Net
![The app](https://github.com/SohamChattopadhyayEE/DRDA-Net/blob/main/videos/Malignant1.gif)
## Objective
- **Dense Residual Dual-shuffle Attention Net (DRDA-Net)** : an efficient framework for breast cancer detection. Overall workflow of the proposed framework is shown below- ![flow diagram](https://github.com/SohamChattopadhyayEE/DRDA-Net/blob/main/figures/Overall%20flow%20diagram.jpg)
- An web application is built to make an end-to-end framework of the DRDA-Net. 

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
## Execution of Training of DRDA-Net
    D:\directory> python train.py
## Execution of The Application Code
- Run the `train.py` and save the trained weights. It is to be noted that the weights has to be saved in a seperated directory nemed "weights". 
- Run `D:\directory> python app.py`. 
- An url similar to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) will be generated which needs to be opened via local browser.
- An interface same as that of the GIF[The app] will appear. 
- There upload the image and click on the `upload` button.  
    
