from Model import model
from Loss import loss


class UniversalFactory:
    classes = []

    def __init__(self, classes=None):
        if classes is not None:
            self.classes = classes
        self.classes_names = {class_.__name__: class_ for class_ in self.classes}

    def create_class(self, class_name, *args, **kwargs):
        instance = self.classes_names[class_name](*args, **kwargs)
        return instance


class ModelFactory(UniversalFactory):
    classes = [
        model.GAVENet
    ]


class LossesFactory(UniversalFactory):
    classes = [
        loss.BCE3Loss,
        loss.RRLoss,
    ]
