class LSTMConfigMIMICIII(object):
    def __init__(
            self,
            total_vocab_size=7012,
            code_vocab_size=6984,
            label_vocab_size=25,
            special_vocab_size=3,
            n_positions=56,
            n_ctx=48,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            batch_size=128,
            epoch=25,
            pos_loss_weight=None,
            lr=1e-4,
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.label_vocab_size = label_vocab_size
        self.special_vocab_size = special_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.epoch = epoch
        self.pos_loss_weight = pos_loss_weight
        self.lr = lr
        
        
class LSTMConfigMIMICIV(LSTMConfigMIMICIII):
    """
    Specialized configuration for MIMIC-IV dataset.
    This class inherits from LSTMConfig but overrides specific values.
    """
    def __init__(self, dataset="mimiciv", dataset_version="2.0"):
        # Call the base class constructor
        super().__init__(dataset, dataset_version)

        # Override parameters specific to MIMIC-IV
        self.total_vocab_size = 25837
        self.code_vocab_size = 25809
        self.n_ctx = 256

class LSTMConfigMIMICIV_ICD9(LSTMConfigMIMICIII):
    """
    Specialized configuration for MIMIC-IV dataset.
    This class inherits from LSTMConfig but overrides specific values.
    """
    def __init__(self, dataset="mimiciv", dataset_version="2.0"):
        # Call the base class constructor
        super().__init__(dataset, dataset_version)
        print(f'Using LSTMConfigMIMICIV_ICD9')
        # Override parameters specific to MIMIC-IV
        self.total_vocab_size = 9100
        self.code_vocab_size = 9072
        self.n_ctx = 192
        
class LSTMConfigeICU(LSTMConfigMIMICIII):
    """
    Specialized configuration for MIMIC-IV dataset.
    This class inherits from LSTMConfig but overrides specific values.
    """

    def __init__(self, dataset="eicu", dataset_version="2.0"):
        # Call the base class constructor
        super().__init__(dataset, dataset_version)

        # Override parameters specific to MIMIC-IV
        self.total_vocab_size = None
        self.code_vocab_size = None
        self.n_ctx = None
