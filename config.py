class CFG:
    train_batch_size = 32
    val_batch_size = 32

    # file paths    
    annotations_fname = "data/image_names_according_to_split/{}_image_list.txt"
    images_dirname = "data/icchipnet_train_val_test/{}/"    

    label_count = 27  # number of vendors in dataset
    train_epochs = 20
    input_shape = 64