
from gguf import GGUFWriter
import torch


if __name__ == '__main__':

    in_file = '/ccn2/u/jwat2002/ccwm/out/CCWM100M_RGB_CF_alldata_bs512/model_00085000.pt'
    out_file = './test.gguf'

    model = torch.load(in_file, weights_only=False)
    config = model['cfg']
    weights = model['weights']


    config['TEST.STRING.ARRAY'] = ('howdy', 'there', 'general', 'kenobi')
    config['TEST.INT.ARRAY'] = (0, 1, 2, 3)

    writer = GGUFWriter(out_file, 'ccwm')
    
    for key, value in config.items():
        if type(value) is str:
            writer.add_string(str(key), value)
        elif type(value) is int:
            writer.add_int64(str(key), value)
        elif type(value) is float:
            writer.add_float32(str(key), value)
        elif type(value) is bool:
            writer.add_bool(str(key), value)
        elif type(value) in [tuple, list]:
            writer.add_array(str(key), value)

    for key, value in weights.items():
        writer.add_tensor(key, value.cpu().numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    writer.close()

