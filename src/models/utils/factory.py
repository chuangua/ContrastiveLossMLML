import logging
logger = logging.getLogger(__name__)
from ..resnet import resnet101
from ..vgg import vgg16

def create_model(args):
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()
    model=resnet101(model_params,num_classes = model_params['num_classes'])
    return model
