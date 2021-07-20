# pseudo-labeling-semseg

This repository contains source code for my graduation thesis: [**Semi-supervised semantic segmentation based on pseudo-labeling**](http://www.zemris.fer.hr/~ssegvic/project/pubs/petrac21ms.pdf)

The model that I'm using for training and evaluation is [SwiftNet](https://github.com/orsic/swiftnet) with ResNet18 as backbone for the recognition encoder.

## Steps to reproduce

### Install requirements
- Python 3.7+

``` 
pip install -r requirements.txt 
```

### Download CamVid
From http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/ download 701 images, as well as their colored masks (about 665 MB in total).

You will also need to download the whole sequences in order to learn with unlabeled images, from [here](http://vis.cs.ucl.ac.uk/Download/G.Brostow/CamVid/). You should only extract frames from sequences 0006R0 and 0016E5, as sequences 01TP and 0005VD are used for testing and evaluating the model. You can use various tools for extracting the frames from videos, e.g. [VLC Media Player](https://www.youtube.com/watch?v=2Lt1lcyweTw). 

Expected dataset structure for CamVid is:
```
data/
    camvid/
        test/
            0001TP_008550.png
            ...
        test_labels/
        
        train/
        train_labels/
      
        unlabeled/
            train/
                # all the unlabeled images
            train_pseudoIt1/
                # all the pseudolabels generated in the first iteration ...
            train_pseudoIt2/
            ...
            
        val/
        val_labels/
```

In order to create masks with 11 semantic classes, you can run `data/mask_maker.py` (you'll need to do this for every subset), just be careful about the path where new images are stored. Those masks should be located in `test_labels`, `train_labels` and `val_labels` prior to training and evaluating the model.

### Training and evaluation

Before running `semisup_cam.py` or `semisup_cif.py`, you'll need to create the following folder structure, so that the model can save and load its best checkpoints while learning:
```
saved_models/
    pseudo_cam/
    pseudo_cifar/

semisup_cam.py
semisup_cif.py
```

All the parameters and hyperparameters can be modified within `semisup_cam.py` and `semisup_cif.py`.

If you have any questions, feel free to contact me via mail: toma.petrac@gmail.com



    
