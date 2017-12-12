from Cnn_class import *
set_path            = "/home/km/Tf_test/CNN_DEV/Cnn/Training_images_set1"
validation_fraction = 0.1
image_size          = 256
image_channels      = 3
classes_labels      = ['fnan', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5']
####CovNet Creation
CNN = ConvNet(img_size_flat=256*256*3,nlayers=5,cll=classes_labels);
####Layer1 Creation
op  = {
        "strides"    :[1, 1, 1, 1],
        "padding"    :'SAME',
        "use_pooling":True,
        "use_relu"   :True,
        "type"       :"conv"
      }; CNN.layer_def(ilayer=1,f_w=3,f_h=3,f_d= 3,n_neuro=32,op=op);
####Layer2 Creation
op  = {
        "strides"    :[1, 1, 1, 1],
        "padding"    :'SAME',
        "use_pooling":True ,
        "use_relu"   :True,
        "type"       :"conv"
      }; CNN.layer_def(ilayer=2,f_w=3,f_h=3,f_d=32,n_neuro=32,op=op);
####Layer3 Creation
op  = {
       "strides"    :[1, 1, 1, 1],
       "padding"    :'SAME',
       "use_pooling":True ,
       "use_relu"   :True,
       "type"       :"conv"
      }; CNN.layer_def(ilayer=3,f_w=3,f_h=3,f_d=32,n_neuro=64,op=op);
####Layer4 Creation
op  = {
       "strides"    :[1, 1, 1, 1],
       "padding"    :'SAME',
       "use_pooling":False,
       "use_relu"   :True,
       "type"       :"full"
      }; CNN.layer_def(ilayer=4,n_neuro=256,op=op);
####Layer5 Creation
op  = {
       "strides"    :[1, 1, 1, 1],
       "padding"    :'SAME',
       "use_pooling":False,
       "use_relu"   :False,
       "type"       :"full"
      }; CNN.layer_def(ilayer=5,n_neuro=7,op=op);
####################
op  = {
    "type"      :"cross_entropy",
    "last_layer":5
   }; CNN.cost_def(op=op)
####################
CNN.model_def()
####################
op  = {
        "type"                    : "Adam"  ,
        "batch_number_saving_rate": 50     ,
        "lr"                      : 1e-4    ,
        "b1"                      : 0.9     ,
        "b2"                      : 0.999   ,
        "ep"                      : 1e-08
      }; CNN.optimizer_def(op=op)
####################
CNN.load_training_data_set(set_path,validation_fraction)
CNN.training_batch_preparation(batch_size=4)
CNN.train(nb_epoch=15)