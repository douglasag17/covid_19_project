import pickle as pkl
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential 


class model(tf.keras.Model):
    """This is the implementation of the TCNN model according to the paper Probabilistic Forecasting
    with Temporal Convolutional Neural Network [https://arxiv.org/pdf/1906.04397.pdf]. There are two
    minor changes here in the decoder; First, we apply a neural network to the encoder vector and
    compute the prediction for the future "w" days (Decoder 1) and then we compute the deviation
    for each day using the exogenous variables (Decoder 2). It is important to highlight that de
    Decoder 1 network has different parameters for each day prediction but the Decoder 2 share
    parameters between days.
    """
    def __init__(self, encoder_params, decoder1_dims):
        """Model initialization
        Parameters
        ----------
        encoder_params : List[Dict]
            A list formed by convolution_parameters dictionaries to instantiate
            the encoder of the model. Each dict contains the information of a Conv1d layer.
        decoder1_dims : List[Int]
            A list with the dims of the decoder 1 Dense layers.
        decoder2_dims : List[Int]
            A list with the dims of the decoder 2 Dense layers.
        n_channels_enc : int
            The number of channels in the encoder.
        """
        super(model, self).__init__()
        #encoder
        self.encoders = []
        for i in range(0,len(encoder_params)):
            encoder = Encoder(encoder_params[i])
            self.encoders.append(encoder)

        #emebeddings
        self.variables_embeddings = temp_embedding = tf.keras.layers.Embedding(input_dim=7, output_dim=1)
        self.flatten = tf.keras.layers.Flatten()
        self.num_1 = decoder1_dims[0] 
        self.num_2 = decoder1_dims[1]
        #Decoder
        self.decoder = Decoder(decoder1_dims[0],decoder1_dims[1])
        
    def call(self, input_tensor):
        """A Pass of the input tensor through the model
        Parameters
        ----------
        input_tensor : Tuple[tf.Tensor]
            A tuple with all the information required to compute a pass through the model.
            input_tensor[0]: The input time series tf.Tensor (Batch_size, input_seq_length,
                                                              n_channels_encoder - n_covariates)
            input_tensor[1]: The input time series days tf.Tensor (Batch_size, input_seq_length)
            input_tensor[2]: The input time series hours tf.Tensor (Batch_size, input_seq_length,
                                                              1)
            input_tensor[3]: The input time series month tf.Tensor (Batch_size, input_seq_length,
                                                              1)
            input_tensor[4]: The output time series days tf.Tensor (Batch_size, output_seq_length,
                                                              1)
            input_tensor[5]: The output time series hours tf.Tensor (Batch_size, output_seq_length,
                                                              1)
            input_tensor[6]: The output time series month tf.Tensor (Batch_size, output_seq_length,
                                                              1)
        Returns
        -------
        tf.Tensor (Batch_size, output_seq_length, 1)
        """
        
        X_1, X_2, days, days_pred  = input_tensor
        embedings = self.variables_embeddings(days)
        pred_embeddings = self.variables_embeddings(days_pred)
        encoder_tensor =  tf.concat([X_1,embedings], axis=-1)
        for encoder in self.encoders:
            encoder_tensor = encoder(encoder_tensor)
        
        enc_output = self.flatten(encoder_tensor[:, -1:, :])
        aux = encoder_tensor.shape[2]
        enc_output.set_shape([None, aux])
        pred_embeddings = tf.squeeze(pred_embeddings, axis=-1)
        pred_embeddings.set_shape([None, 7])
        output = self.decoder(enc_output, pred_embeddings)        
        outputs = tf.keras.backend.expand_dims(output, axis=1)
        outputs = tf.math.exp(outputs)
        outputs = tf.reshape(outputs, [-1,7,self.num_2 // 7])
        X_2 = tf.slice(X_2, [0,9,0], [-1,-1,self.num_2 // 7])
        outputs = tf.math.multiply(outputs,X_2)
        return outputs


class Encoder(tf.keras.Model):
    
        def __init__(self, conv1_params):
            """Initialization of the two convoutional layers that compose the model.
            It receives two dicts with the parameters of the 1D convolution layers.
            To be consistent with the TCNN implementation, the padding of the
            convolutions must be 'causal' and the dilation_rate argument
            must be provided for each layer.
            Parameters
            ----------
            conv1_params : dict
                The parameters for the first convolutional layer of the residual block of the TCNN
            conv2_params : dict
                The parameters for the second convolutional layer of the residual block of the TCNN
            """
            super(Encoder, self).__init__()
            self.conv1 = tf.keras.layers.Conv1D(**conv1_params)
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.relu_1 = tf.keras.layers.ReLU()
    
        def call(self, x):
            aux_int = self.conv1(x)
            aux_int = self.bn1(aux_int)
            y = self.relu_1(aux_int)
            return y

class Decoder(tf.keras.Model):
    
        def __init__(self, units_1, units_2):
            """Initialization of the two convoutional layers that compose the model.
            It receives two dicts with the parameters of the 1D convolution layers.
            To be consistent with the TCNN implementation, the padding of the
            convolutions must be 'causal' and the dilation_rate argument
            must be provided for each layer.
            Parameters
            ----------
            conv1_params : dict
                The parameters for the first convolutional layer of the residual block of the TCNN
            conv2_params : dict
                The parameters for the second convolutional layer of the residual block of the TCNN
            """
            super(Decoder, self).__init__()
            self.units = units_1
            self.dense_0 = tf.keras.layers.Dense(units_1)
            self.dense1 = tf.keras.layers.Dense(units_1)
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.relu = tf.keras.layers.ReLU()
            self.dense2 = tf.keras.layers.Dense(units_2)
            self.relu_2 = tf.keras.layers.ReLU()
            self.bn2 = tf.keras.layers.BatchNormalization()
    
        def call(self, x, y):
            #x.set_shape([None,])
            print(x, y)
            aux_int = self.dense1(x)
            aux_int_2 = self.dense_0(y)
            res_1 =  self.bn1(tf.concat([aux_int, aux_int_2], axis=-1))
            res_1 = self.relu(res_1)
            y = self.dense2(res_1)
            return y

def get_model(weigths_path, forecast_window, training_window, num_dep):
    dilation_rates=[1,2,2,4,4,6,6]
    kernel_size=2
    number_features=256
    enc_convolution_params = []
    for idx, dilation_rate in enumerate(dilation_rates):
        conv_params = dict()
        if idx == 0:
            conv_params["input_shape"] = (forecast_window, num_dep)
        conv_params["filters"] = number_features
        conv_params["kernel_size"] = kernel_size
        conv_params["padding"] = "causal"
        conv_params["dilation_rate"] = dilation_rate
        enc_convolution_params.append(conv_params)
    net = model(enc_convolution_params, [50,num_dep*forecast_window])
    #aux = tf.constant(1, tf.float64,  shape=(1,10,33))
    #net((aux, aux))
    #net.load_weights(weigths_path)
    return net