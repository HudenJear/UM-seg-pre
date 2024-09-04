from .caclass_model import ClassModel
from .sequential_model import SequentialModel
from .aux_model import AuxModel,ApartModel
from .assist_model import AssistModel
from .logger_utils import get_root_logger
from copy import deepcopy

def build_model(opt):
  """Build model from options.

  Args:
      opt (dict): Configuration. It must contain:
          model_type (str): Model type.
  """
  model_type = opt.pop('model_type')
  opt = deepcopy(opt)
  if model_type in ['ClassModel',"SequentialModel","AuxModel","AssistModel","ApartModel"]:
    if model_type == 'ClassModel':
      model = ClassModel(opt)
    elif model_type == 'SequentialModel':
      model = SequentialModel(opt)
    elif model_type == 'AuxModel':
      model = AuxModel(opt)
    elif model_type == 'AssistModel':
      model = AssistModel(opt)
    elif model_type == 'ApartModel':
      model = ApartModel(opt)
    
      
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
  else: 
    model =None
    logger = get_root_logger()
    logger.info('Model '+model_type+' is NOT created. No matched name.')
  
  return model