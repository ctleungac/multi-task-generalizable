import tensorflow as tf
import tensorflow.keras.backend as K

class getMIOutput(tf.keras.callbacks.Callback):
    def __init__(self, trn, num_selection, embedding_network,target_layer, do_save_func=None, *kargs, **kwargs):
        super(getMIOutput, self).__init__(*kargs, **kwargs)
        self.layer_values = []
        self.trn = trn
        self.datalist = []
        self.num_selection = num_selection
        self.embedding_network = embedding_network
        self.target_layer = target_layer
        self.do_save_func = do_save_func
        
    def on_train_begin(self, logs={}):
        self.layer_values = []
        self.layerixs = []
        self.layerfuncs = []

        # Assuming the embedding layer index is 3, change it based on the actual index
        embedding_layer_index = self.target_layer

        for lndx, l in enumerate(self.embedding_network.layers):
            self.layerixs.append(lndx)
            self.layer_values.append(lndx)
            self.layerfuncs.append(K.function(self.embedding_network.inputs, [l.output,]))

    def on_epoch_end(self, epoch, logs={}):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            return
        
        data = {
            'activity_tst': []    # Activity in each layer for the test set
        }

        for lndx, layerix in enumerate(self.layerixs):
            if (lndx == self.target_layer):  # Assuming you want to access a specific layer
                data['activity_tst'].append(self.layerfuncs[lndx]([self.trn[:self.num_selection],])[0])

        save_dic = {'epoch': epoch, 'data': data}
        self.datalist.append(save_dic)

