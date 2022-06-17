from torch import nn
from typing import Callable

from gem.pipelines.common import BasicAugmentation


class CrossEntropyClassifier(BasicAugmentation):
    """ Standard cross-entropy classification as baseline.

    See `BasicAugmentation` for a documentation of the available hyper-parameters.
    """

    def get_loss_function(self) -> Callable:

        return nn.CrossEntropyLoss(reduction='mean')


    @staticmethod
    def get_pipe_name():

        return 'xent'
