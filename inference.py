import torch
from external.AcademicCodec.quantization  import ResidualVectorQuantizer
from external.AcademiCodec.Encodec_24k_240d.model import Encodec

PATH = '/home/dev/other/AcademicCodec/checkpoint.pth'

model = Encodec(n_filters=32, D=512, ratios=[6, 5, 4, 2]) 
parameter_dict = torch.load(PATH)
new_state_dict = OrderedDict()
for k, v in parameter_dict.items(): 
    name = k[7:] 
    new_state_dict[name] = v
model.load_state_dict(new_state_dict) 
