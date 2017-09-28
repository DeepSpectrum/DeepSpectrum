

class TensorFlowExtractor():
    def __init__(self, net, weights_path):
        from deep_spectrum_extractor import models

        net = models.load_model(net)
        data_spec = models.get_data_spec(model_instance=net)
