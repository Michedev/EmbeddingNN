from typing import List, Union
import numpy as np
import keras

class EmbeddingNN:

    loss=None
    """
    Type: a string or a loss function from keras.losses
    The loss function to optimize
    """
    optimizer=None
    """
    :type A string or an optimizer object from keras.optimizers
    The optimizer of the neural network
    """
    batch_norm_first: bool=None
    """
    :type: bool
    If true add a batch normalization layer in the first layer
    of the quantitative nn
    """

    quantitative_layers: List=None
    """
    :type: List of layers
    The list of layers having in input the quantitative variables
    
    """
    concat_layers=None
    """
    :type: List of layers
    The list of layers having in input the embedded vectors and
    the quantitative variables
    """
    gausnoise_first: bool=None
    """
    :type: boolean
    If true add gaussian noise at the first level
    """
    gausnoise_stdev: float=None
    """
    The standard devation of the gaussian noise.
    It has effect only if gausnoise_first is True
    """

    fithist=None
    """
    :type List of keras History object
    contains all the fit histories
    """

    def fit(self, X: np.ndarray, y: np.array, catcols: List[int], quacols=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0., validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, **kwargs):
        """
            X: Numpy array of training data (if the model has a single input),
                or list of Numpy arrays (if the model has multiple inputs).
                If input layers in the model are named, you can also pass a
                dictionary mapping input names to Numpy arrays.
                `x` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            y: Numpy array of target (label) data
                (if the model has a single output),
                or list of Numpy arrays (if the model has multiple outputs).
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
                `y` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            catcols: List of integers
                list of column indexes that contains categorical variables
            quacols: List of integers or None
                list of columns indexes that contains quantitative variables.
                If not specified it will be set as the remaining columns from
                X subtracting the catcols
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling.
            validation_data: tuple `(x_val, y_val)` or tuple
                `(x_val, y_val, val_sample_weights)` on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
            validation_steps: Only relevant if `steps_per_epoch`
                is specified. Total number of steps (batches of samples)
                to validate before stopping.
        """
        pass


    def _embeds_nn(self, X: np.ndarray, catcols: List[int], embed_size: Union[int, List[int]]):
        pass

    def _quant_nn(self, n_quantcols):
        """
        Create the layers for quantitative variables
        :param n_quantcols: number of quantitative variables
        :return: a tuple composed by the input layer and
                 the last layer of the nn
        """
        pass

    def _concat_nn(self, qualayer: keras.layers.Layer, *embedlayers: List[keras.layers.Layer]):
        """
        Create the part of the nn that receive in input the quantitative part and the embeds part
        :param qualayer: the last layer of nn that have in input the quantitative variables
        :param embedlayers: the last layers of any embed
        :return: the output layer
        """
        pass

    def _split_matrix(self, X: np.ndarray, catcols: List[int], quacols: List[int]):
        """
        Split the X in a dictionary that map each input name
        to the respective array/matrix of values.
        The nomenclature of each key of the dictionary is established
        internally
        :param X: The matrix to split
        :param catcols: the index of the columns that contains the categorical variables
        :param quacols: the index of the columns that contains the quantitative variables
        :return: a dictionary that map each input layer name to the respective values
        """
        pass