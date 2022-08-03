# ContrastiveLossMLML
This repository contains the source code for the experiments of the article

    "Label Structure Preserving Contrastive Embedding for Multi-Label Learning with Missing Labels" 
    Zhongchen Ma†, Lisha Li†, Qirong Mao, Senior Member, IEEE, and Songcan Chen∗, Senior Member, IEEE
    
    
you can try:
    
    $ python train_clml.py --dataset ./dataset/coco_train_0.75left.txt --data /home/mscoco --b 64 --loss BCE --lambda_ 1.00 --useclml True --threshold 0.8


and then start training:
    
    creating model...
    num_classes =  80
    model use imagenet pretained!
    done

    loading annotations into memory...
    Done (t=4.78s)
    creating index...
    index created!
    load class_nums =  80
    len(val_dataset)):  40137
    len(train_dataset)):  82081
    Epoch [0/80], Step [000/1283], LR 4.0e-06, Loss: 3719.9
    Epoch [0/80], Step [100/1283], LR 4.0e-06, Loss: 2793.5
    ···
