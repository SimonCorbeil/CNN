import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import glob
import cv2
import os

class ConvNet():
    """
    ########################################
    ##############ConvNet####################
    #Class designed to construct, train,
    #valid and use a ConvNet
    #1 - img_size_flat : size of the vector
    #    containing an unfolded image
    #    i.e. wide * height * depth
    #2 - nlayers       : number of layers,
    #    including inputs , hidden and
    #    outputs layers
    #3 - cll           : classes labels
    ########################################
    ########################################
    """
    def __init__(
                  self              ,
                  img_size_flat     ,
                  nlayers=1         ,
                  cll    = ['0','1']
                ):

        self.imgcha  =  3;                                                # number of channel in the inpup images (depth)
        self.w       = {};                                                # dictionary containing the weights
        self.b       = {};                                                # dictionary containing the biais
        self.l       = {};                                                # dictionary containing the layers
        self.nlayers = nlayers                                            # numbers of layers in the neural net including convolutional
                                                                          # and fully connected layer
        self.imgsize         = int(np.sqrt(img_size_flat / self.imgcha))  # wide or heigth of the image
        self.imgsizeflat     = img_size_flat                              # length of the input image:wide*height*dep
        self.errorevolution  = []                                         # list of batch index, training and validation error
        self.savedmodelpath  = "./model/"
        self.modelname       = "model"
        self.classeslabel    = cll                                        # list of labels
        self.numclasses      = len(cll)                                   # number of classes
        self.classesindex    = {}                                         # dictionnary linking label and index
        for I in range(self.numclasses): self.classesindex.update({self.classeslabel[I]:I})

        # Placeholders for input images

        # tensorflow placeholder for flatten image input
        self.x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='xflat')
        # tensorflow placeholder for intial image input
        self.ximg = tf.reshape(self.x, [-1, self.imgsize, self.imgsize, self.imgcha], name='xcube')

        # Placeholders for labels and predictions model

        # tensorflow placeholder for image softmax probabilities associated to the classification
        self.ysofttrue = tf.placeholder(tf.float32, shape=[None, self.numclasses], name='y_soft_true')
        # tensorflow placeholder for image softmax probabilities prediction associated to the classification
        self.ysoftpred = tf.placeholder(tf.float32, shape=[None, self.numclasses], name='y_soft_pred')
        # tensorflow placeholder for image label associated to the classification
        self.yhardtrue = tf.argmax(self.ysofttrue, axis=1, name='y_hard_true')
        # tensorflow placeholder for image label associated to the classification
        self.yhardpred = tf.argmax(self.ysoftpred, axis=1, name='y_hard_pred')


        # Creation of dictinairy for all Variables (weight, biases and layers)

        for l in range(self.nlayers): self.w.setdefault("w" + str(l))
        for l in range(self.nlayers): self.b.setdefault("b" + str(l))
        for l in range(self.nlayers): self.l.setdefault("l" + str(l))

    ########## definition and structure of the ConNet #############
    def layer_def(
                   self     ,
                   ilayer =1,
                   f_w    =1,
                   f_h    =1,
                   f_d    =1,
                   n_neuro=1,
                   op={
                        "strides": [1, 1, 1, 1],
                        "padding":       'SAME',
                        "use_pooling":    False,
                        "use_relu":        True,
                        "type":           "conv"
                      }
                 ):
        """
        #########################################
        ############## layer_def ################
        #layer constructor
        #1 - ilayer   : index of the layer
        #2 - f_w      : wide of the filter
        #3 - f_h      : height
        #4 - f_d      : depth of the filter
        #5 - n_neuro  : number of neurons
        #              (or number of filter)
        #6 - op       : dictionnary of layers
        #               details
        #    including: "strides","padding",
        #    use_pooling","use_relu","type"
        #########################################
        #########################################
        """
        # if the layer is a convolutional layer and the first one
        if (ilayer == 1  and op["type"] == "conv"):
            f_shape = [f_w, f_h, f_d, n_neuro]
            self.w["w" + str(ilayer)] = tf.Variable(tf.truncated_normal(f_shape, stddev=0.05))
            self.b["b" + str(ilayer)] = tf.Variable(tf.constant(0.05, shape=[n_neuro]))
            Input = self.ximg
            w = self.w["w" + str(ilayer)]
            b = self.b["b" + str(ilayer)]
            l = tf.nn.conv2d(input=Input, filter=w, strides=op["strides"], padding=op["padding"]) + b
            if op["use_pooling"]: l = tf.nn.max_pool(value=l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if op["use_relu"]: l = tf.nn.relu(l)
            self.l["l" + str(ilayer)] = l
        # if the layer is a convolutional layer hidden one
        elif (ilayer > 1 and op["type"] == "conv"):
            f_shape = [f_w, f_h, f_d, n_neuro]
            self.w["w" + str(ilayer)] = tf.Variable(tf.truncated_normal(f_shape, stddev=0.05))
            self.b["b" + str(ilayer)] = tf.Variable(tf.constant(0.05, shape=[n_neuro]))
            Input = self.l["l" + str(ilayer - 1)]
            w = self.w["w" + str(ilayer)]
            b = self.b["b" + str(ilayer)]
            l = tf.nn.conv2d(input=Input, filter=w, strides=op["strides"], padding=op["padding"]) + b
            if op["use_pooling"]: l = tf.nn.max_pool(value=l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if op["use_relu"]: l = tf.nn.relu(l)
            self.l["l" + str(ilayer)] = l
        # if the layer is a fully connected layers
        elif (ilayer > 1 and op["type"] == "full"):
            Input = self.l["l" + str(ilayer - 1)]
            Input_shape = Input.get_shape()
            Input_size = Input_shape[1:4].num_elements()
            Input_flat = tf.reshape(Input, [-1, Input_size])
            f_shape = [Input_size, n_neuro]
            self.w["w" + str(ilayer)] = tf.Variable(tf.truncated_normal(f_shape, stddev=0.05))
            self.b["b" + str(ilayer)] = tf.Variable(tf.constant(0.05, shape=[n_neuro]))
            w = self.w["w" + str(ilayer)]
            b = self.b["b" + str(ilayer)]
            l = tf.matmul(Input_flat, w) + b
            if op["use_relu"]:    l = tf.nn.relu(l)
            self.l["l" + str(ilayer)] = l

    def model_def(
                   self                         ,
                   model_name       = "model"   ,
                   saved_model_path = "./model/"
                 ):
        """
        ########################################
        ############## model_def ################
        #definition of the predictive model
        #1 - last_layer : label of the last layer
        ########################################
        ########################################
        """
        last_layer          = self.l["l" + str(self.nlayers)]
        self.modelname      = model_name
        self.savedmodelpath = saved_model_path
        self.ysoftpred      = tf.nn.softmax(last_layer,         name='y_soft_pred')
        self.yhardpred      = tf.argmax(self.ysoftpred, axis=1, name='y_hard_pred')
        tf.add_to_collection("y_soft_pred", self.ysoftpred)
        tf.add_to_collection("y_hard_pred", self.yhardpred)

    def cost_def(
                  self,
                  op={
                      "type"      : "cross_entropy",
                      "last_layer": 1
                     }
                ):
        """
        ########################################
        ############## cost_def ################
        #definition of the cost function
        #1 - op : dictionary of cost function
        #         details including "type" cost
        #         cost function and the index
        #         of the "last layer"
        ########################################
        ########################################
        """
        self.nllayer       = op["last_layer"]
        last_layer         = self.l["l" + str(op["last_layer"])]
        y_soft_true        = self.ysofttrue
        self.crossentropy  = tf.nn.softmax_cross_entropy_with_logits(logits=last_layer, labels=y_soft_true)
        self.cost          = tf.reduce_mean(self.crossentropy)

    def optimizer_def(
                       self     ,
                       op={
                           "type"                     : "Adam"  ,
                           "batch_number_saving_rate" : 1       ,
                           "lr"                       : 1e-4    ,
                           "b1"                       : 0.9     ,
                           "b2"                       : 0.999   ,
                           "ep"                       : 1e-08
                          }
                     ):
        """
        ########################################
        ############## optimizer_def ###########
        #1 - op       :options of the optimizer
        #    include: "type" of solver ("Adam"),
        #             "batch_number_saving_rate"
        #             and for now just the Adam
        #             algo  parameters:
        #             "lr" (learning rate)
        #             "b1" (beta1)
        #             "b2" (beta2)
        #             "ep" (epsilon)
        ########################################
        ########################################
        """
        isexitmodeldir        = os.path.isdir(self.savedmodelpath)
        self.saver            = tf.train.Saver()
        self.savingrate       = op["batch_number_saving_rate"]
        self.optimizertype    = op["type"]
        self.learning_rate    = op["lr"]

        if (not isexitmodeldir): os.mkdir(self.savedmodelpath)

        if( op["type"] == "Adam"):
            self.optimizer_beta1  = op["b1"]
            self.optimizer_beta2  = op["b2"]
            self.epsilon          = op["ep"]
            self.optimizer        = tf.train.AdamOptimizer(
                                                            learning_rate= self.learning_rate  ,
                                                            beta1        = self.optimizer_beta1,
                                                            beta2        = self.optimizer_beta2,
                                                            epsilon      = self.epsilon
                                                           ).minimize(self.cost)
    ########## loading methods  #############
    def load_training_data_set(
                                self               ,
                                set_path           ,
                                validation_fraction
                              ):
        """
        ########################################
        ######## load_training_data_set ########
        # 1 - set_path : path of the images
        # directories, it must contain a folder
        # for each images classes, those
        # directories must be named with the
        # name of the classes
        # 2 - validation_fraction : part of the
        # total images set reserved for the
        # validation
        ########################################
        ########################################
        """
        self.setpath  = set_path
        image_size    = self.imgsize
        dir_label     = os.listdir(self.setpath)
        vf            = validation_fraction
        images_tr     = [];
        labels_tr     = [];
        images_vl     = [];
        labels_vl     = [];
        #loop over all directories (each of them containing one class of image)
        for c in dir_label:
            path         = os.path.join(self.setpath, c + '/*g')
            index        = self.classesindex[c]
            files        = glob.glob(path)
            nb_tr_images = 0
            # loop over all the images of a given class
            for fl in files:
                trainsetsize = int((1.0 - vf) * len(files))
                label = np.zeros(self.numclasses) #vectorization class label
                image = cv2.imread(fl)
                ##################
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0 / 255.0)
                ##################
                image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
                label[index] = 1.0
                # if the training size is not reached yet
                if (nb_tr_images < trainsetsize):
                    images_tr.append(image)
                    labels_tr.append(label)
                    nb_tr_images += 1
                # if the it is time of feeding the validation set
                else:
                    images_vl.append(image)
                    labels_vl.append(label)
            print("nb training images             : ", len(images_tr))
            print("nb validation images           : ", len(images_vl))
            print("validation set / total images  : ", int(100 * (len(images_vl) / (len(images_tr) + len(images_vl)))), "%")
        self.nbtrimages      = len(images_tr)
        self.nbvlimages      = len(images_vl)
        self.traindata       = np.array(images_tr)
        self.trainlabel      = np.array(labels_tr)
        self.validationdata  = np.array(images_vl)
        self.validationlabel = np.array(labels_vl)

    def load_new_images(
                         self         ,
                         set_path='./'
                       ):
        """
        ########################################
        ######## load_new_images ###############
        # 1 - set_path : path to the directory
        #     containing the images files
        ########################################
        ########################################
        """
        self.setpath  = set_path
        image_size    = self.imgsize
        images        = []
        path          =  os.path.join(set_path + '/*')
        list_of_files = glob.glob(path)
        for fl in list_of_files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            ###################
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            ##################
            images.append(image)
        self.newimages = np.array(images)

    def load_model(
                    self                        ,
                    saved_model_path = "./model",
                    model_name       = "model"
                  ):
        """
        ########################################
        ############### load_model #############
        # 1 - saved_model_path
        # 2 - model_name
        ########################################
        ########################################
        """
        self.saver = tf.train.import_meta_graph(saved_model_path + "/" + model_name + '.meta')

    ########## training and prediction methods #############
    def training_batch_preparation(
                                    self        ,
                                    batch_size=1
                                  ):
        """
        ########################################
        ##### training_batch_preparation #######
        # 1 - batch_size : number of image in a
        #     batch
        ########################################
        ########################################
        """
        self.randindextrainingimage = []
        self.nbtrainingbatch        = 0
        self.batchsize              = batch_size
        self.nbimagestrset          = np.shape(self.traindata)[0]
        self.nbbatchperepoch        = int(self.nbimagestrset / self.batchsize)
        self.randindextrainingimage = np.random.permutation(self.nbimagestrset)

    def training_batch_distribution(
                                     self      ,
                                     ibatch = 0
                                   ):
        """
        ########################################
        ######## training_batch_distribution ###
        # return a random selection of image as
        # batch for validation
        # 1 - ibatch : index of the actual batch
        ########################################
        ########################################
        """
        np.random.seed(seed=int(time.time()))
        nbs = self.batchsize
        neb = self.nbbatchperepoch
        while(ibatch >= neb): ibatch = ibatch - neb
        I = (ibatch + 0) * nbs
        J = (ibatch + 1) * nbs
        Index = self.randindextrainingimage[I:J]
        batch_image = self.traindata[Index]
        batch_label = self.trainlabel[Index]
        return (batch_image, batch_label)

    def validation_batch_distribution(
                                       self,
                                     ):
        """
        ########################################
        ##### validation_batch_distribution ####
        # return a random selection of images as
        # batch for validation
        ########################################
        ########################################
        """
        np.random.seed(seed=int(time.time()))
        nbs = self.batchsize
        nbv = self.nbvlimages
        assert nbs < nbv
        rand_index_image = np.random.permutation(nbv)
        Index       = rand_index_image[0:nbs]
        batch_image = self.validationdata[Index]
        batch_label = self.validationlabel[Index]
        return (batch_image, batch_label)

    def optimize(
                  self    ,
                  session ,
                  nb_batch
                ):
        """
        ########################################
        ############## optimize ################
        #1 - session  : tensorflow session
        #2 - nb_batch : number of batch for the
        #               training process
        ########################################
        ########################################
        """
        nb_Epoch   = int(nb_batch/self.nbbatchperepoch)
        path       = self.savedmodelpath
        modelname  = self.modelname
        savingrate = self.savingrate  #each savingrate number of batch, results are saved
        for i in range(nb_batch):

            # new batch for training : ximg image and y label
            (x_batch, y_true_batch) = self.training_batch_distribution(ibatch=i)
            ximg_batch              = x_batch
            x_batch                 = x_batch.reshape(self.batchsize, self.imgsizeflat)
            # inputs tensorflow formating to feed the optimizer
            dict_train              = {self.x: x_batch, self.ysofttrue: y_true_batch}
            session.run(self.optimizer, feed_dict=dict_train)
            self.nbtrainingbatch += 1

            # computation of prediction in the actual DNN training state
            #last_layer = self.l["l" + str(self.nlayers)]
            #self.ysoftpred = tf.nn.softmax(last_layer,         name='y_soft_pred')
            #self.yhardpred = tf.argmax(self.ysoftpred, axis=1, name='y_hard_pred')

            # saving block
            if ((i) % (savingrate) == 0):
                self.saver.save(session, path + modelname, global_step=i)
                tloss = self.loss_eval(session, ximg_batch   , y_true_batch)
                (ximg_vl_batch, y_vl_true_batch) = self.validation_batch_distribution()
                vloss = self.loss_eval(session, ximg_vl_batch, y_vl_true_batch)
                print("Epoch: ", int(i/self.nbbatchperepoch) + 1,"/",nb_Epoch)
                print("Batch: ", i,)
                print("actual training   loss:", tloss)
                print("actual validation loss:", vloss)
                self.errorevolution.append([i,tloss,vloss])

    def loss_eval(
                   self        ,
                   session     ,
                   ximg_batch  ,
                   y_true_batch
                 ):
        """
        ########################################
        ############## loss_eval ###############
        #definition of the valuation of loss
        #for input images set
        # 1 - session      : tf.Session
        #                    instantiation
        # 2 - ximg_batch   : set of images input
        #                    not flat format
        # 3 - y_true_batch : set true images
        #                    labels
        ########################################
        ########################################
        """
        last_layer    = session.run(self.l["l" + str(self.nlayers)],{self.ximg:ximg_batch})
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=last_layer, labels=y_true_batch)
        cost          = session.run(cross_entropy)
        loss          = session.run(tf.reduce_mean(cost))
        return loss

    def train(
               self      ,
               nb_epoch=1
             ):
        """
        ########################################
        ############### train ##################
        # launch the training process:
        # start a tf session and run optimize
        # the number of times required
        # 1 - nb_epoch : number of epochs for
        #     the training
        ########################################
        ########################################
        """
        with tf.Session() as session:
            nb_tr_images       = self.nbtrimages
            batch_size         = self.batchsize
            nb_batch_per_epoch = int(nb_tr_images / batch_size)
            nb_batch           = nb_epoch * nb_batch_per_epoch
            session.run(tf.global_variables_initializer())
            self.optimize(session, nb_batch)

    def predict(
                 self                        ,
                 input_data_path  = "./"     ,
                 saved_model_path = "./model",
                 model_name       = "model"  ,
                 pred_number      = 1
               ):
        """
        ########################################
        ############### predict ################
        # 1 - input_data_path : path of the
        #     directory containing the images
        # 2 - saved_model_path: path of the
        #     trained model
        # 3 - model_name : name of the model
        #     used for prediction
        # 4 - pred_number: number of images to label
        ########################################
        ########################################
        """
        path          = saved_model_path + "/" + model_name
        self.graph    = tf.Graph()
        self.session  = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # restoration of saved model
            self.saver    = tf.train.import_meta_graph(path + '.meta',clear_devices=True)
            self.saver.restore(self.session, path)

            # loading of new unlabeled images
            self.load_new_images(input_data_path)
            new_images    = self.newimages

            # getting predictors
            softpred = tf.get_collection('y_soft_pred')
            hardpred = tf.get_collection('y_hard_pred')

            # show resulting prediction and associated images
            for I in new_images[0:pred_number]:
                fig,axes        = plt.subplots(1)
                Irs = np.reshape(I, [1, self.imgsize, self.imgsize, self.imgcha])
                axes.imshow(I, vmin=self.imgsize, vmax=self.imgsize, interpolation='nearest', cmap='seismic')
                arg = self.session.run(hardpred, {"xcube:0": Irs})[0]
                nf = self.classeslabel[arg[0]]
                print("Number of fingers:", nf)
                plt.show()