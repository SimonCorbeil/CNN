from Cnn_class import *
####CovNet

nbpredimg      = 50
model_path     = "./model"
model_name     = "model-6250"
set_path       = "/home/km/Tf_test/CNN_DEV/Cnn/Test_images_set1"
classes_labels = ['fnan', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5']
CNN = ConvNet(img_size_flat=256*256*3,nlayers=5,cll=classes_labels);
CNN.predict(input_data_path=set_path,saved_model_path=model_path,model_name=model_name,pred_number=nbpredimg)
