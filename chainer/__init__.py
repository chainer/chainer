import pkg_resources

from chainer import dataset
from chainer import flag
from chainer import function
from chainer import function_set
from chainer.functions import basic_math
from chainer import link
from chainer import optimizer
from chainer import serializer
from chainer import trainer
from chainer import variable


__version__ = pkg_resources.get_distribution('chainer').version

AbstractSerializer = serializer.AbstractSerializer
Chain = link.Chain
ChainList = link.ChainList
Deserializer = serializer.Deserializer
Dataset = dataset.Dataset
Flag = flag.Flag
Function = function.Function
FunctionSet = function_set.FunctionSet
GradientMethod = optimizer.GradientMethod
Link = link.Link
Optimizer = optimizer.Optimizer
Serializer = serializer.Serializer
Trainer = trainer.Trainer
Variable = variable.Variable

create_standard_trainer = trainer.create_standard_trainer

ON = flag.ON
OFF = flag.OFF
AUTO = flag.AUTO

basic_math.install_variable_arithmetics()
