import json
import yaml
import os
from collections import OrderedDict

def load_opt(in_path):
    opt = yaml.safe_load(open(in_path))
    # opt['model']['lr'] = opt['lr']
    return opt

def save_opt(save_path, opt):
    with open(save_path, 'w') as outfile:
        yaml.dump(opt, outfile, default_flow_style=False, sort_keys=False)

# def save_opt(save_path, opt):
#     opt_dict = {}
#     for k, v in sorted(vars(opt).items()):
#         opt_dict[k] = v

#     with open(save_path, 'wt') as f:
#         json.dump(opt_dict, f, indent=4)
#         print("options saved to:", save_path)

def make_output_folders(root_dir, folder_names):
    # remove nohup.out 
    nohup_path = os.path.join(root_dir, "../../nohup.out")
    if os.path.exists(nohup_path): 
        os.remove(nohup_path)
        print("nohup.out deleted:", nohup_path)

    dirs = {}
    for name in folder_names:
        odir = os.path.join(root_dir, name)
        os.makedirs(odir, exist_ok=True)
        name_key = name.split('_')[0]
        dirs[name_key] = odir
    return dirs