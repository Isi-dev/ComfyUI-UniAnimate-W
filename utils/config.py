import os
import yaml
import json
import copy
import argparse

from ..utils import logging
# logger = logging.get_logger(__name__)

class Config(object):
    def __init__(self, config_file, load=True, cfg_dict=None, cfg_level=None):
        self._level = "cfg" + ("." + cfg_level if cfg_level is not None else "")
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        self.config_file_loc = os.path.join(parent_directory, config_file)

        if load:
            self.args, self.unknown_args = self._parse_args()
            # logger.info("Loading config from {}.".format(self.args.cfg_file))
            self.need_initialization = True
            # cfg_base = self._load_yaml(self.args) # self._initialize_cfg()
            cfg_dict = self._load_yaml(self.args)
            # cfg_dict = self._merge_cfg_from_base(cfg_base, cfg_dict)
            cfg_dict = self._update_from_args(cfg_dict)
            self.cfg_dict = cfg_dict
        self._update_dict(config_file, cfg_dict)
    
    def _parse_args(self):
        parser = argparse.ArgumentParser(
            description="Argparser for configuring the codebase"
        )
        parser.add_argument(
            "--cfg",
            dest="cfg_file",
            help="Path to the configuration file",
            default= self.config_file_loc
        )
        parser.add_argument(
            "--init_method",
            help="Initialization method, includes TCP or shared file-system",
            default="tcp://localhost:9999",
            type=str,
        )
        parser.add_argument(
            '--debug',
            action='store_true', 
            default=False, 
            help='Output debug information'
        )
        parser.add_argument(
            '--windows-standalone-build',
            action='store_true', 
            default=False, 
            help='Indicates if the build is a standalone build for Windows'
        )



        # New Command Line Arguments
        parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0", help="Specify the IP address to listen on (default: 127.0.0.1). If --listen is provided without an argument, it defaults to 0.0.0.0. (listens on all)")
        parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
        parser.add_argument("--tls-keyfile", type=str, help="Path to TLS (SSL) key file. Enables TLS, makes app accessible at https://... requires --tls-certfile to function")
        parser.add_argument("--tls-certfile", type=str, help="Path to TLS (SSL) certificate file. Enables TLS, makes app accessible at https://... requires --tls-keyfile to function")
        parser.add_argument("--enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*", help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
        parser.add_argument("--max-upload-size", type=float, default=100, help="Set the maximum upload size in MB.")

        parser.add_argument("--extra-model-paths-config", type=str, default=None, metavar="PATH", nargs='+', action='append', help="Load one or more extra_model_paths.yaml files.")
        parser.add_argument("--output-directory", type=str, default=None, help="Set the ComfyUI output directory.")
        parser.add_argument("--temp-directory", type=str, default=None, help="Set the ComfyUI temp directory (default is in the ComfyUI directory).")
        parser.add_argument("--input-directory", type=str, default=None, help="Set the ComfyUI input directory.")
        parser.add_argument("--auto-launch", action="store_true", help="Automatically launch ComfyUI in the default browser.")
        parser.add_argument("--disable-auto-launch", action="store_true", help="Disable auto launching the browser.")
        parser.add_argument("--cuda-device", type=int, default=None, metavar="DEVICE_ID", help="Set the id of the cuda device this instance will use.")
        parser.add_argument("--force-channels-last", action="store_true", help="Force channels last format when inferencing the models.")

        parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1, help="Use torch-directml.")

        parser.add_argument("--disable-ipex-optimize", action="store_true", help="Disables ipex.optimize when loading models with Intel GPUs.")
        parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")
        parser.add_argument("--default-hashing-function", type=str, choices=['md5', 'sha1', 'sha256', 'sha512'], default='sha256', help="Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.")

        parser.add_argument("--disable-smart-memory", action="store_true", help="Force ComfyUI to agressively offload to regular ram instead of keeping models in vram when it can.")
        parser.add_argument("--deterministic", action="store_true", help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.")

        parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
        parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")
        # parser.add_argument("--windows-standalone-build", action="store_true", help="Windows standalone build: Enable convenient things that most people using the standalone windows build will probably enjoy (like auto opening the page on startup).")

        parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata in files.")
        parser.add_argument("--disable-all-custom-nodes", action="store_true", help="Disable loading all custom nodes.")

        parser.add_argument("--multi-user", action="store_true", help="Enables per-user storage.")

        parser.add_argument("--verbose", action="store_true", help="Enables more debug prints.")
        # The default built-in provider hosted under web/
        DEFAULT_VERSION_STRING = "comfyanonymous/ComfyUI@latest"

        parser.add_argument(
            "--front-end-version",
            type=str,
            default=DEFAULT_VERSION_STRING,
            help="""
            Specifies the version of the frontend to be used. This command needs internet connectivity to query and
            download available frontend implementations from GitHub releases.

            The version string should be in the format of:
            [repoOwner]/[repoName]@[version]
            where version is one of: "latest" or a valid version number (e.g. "1.0.0")
            """,
        )
        # End of New Command Line Arguments

        
        parser.add_argument(
            "opts",
            help="Other configurations",
            default=None,
            nargs=argparse.REMAINDER
        )


        args, unknown = parser.parse_known_args()

        print(f"Unrecognized args: {unknown}")

        return args, unknown
        # return parser.parse_args()


    def _path_join(self, path_list):
        path = ""
        for p in path_list:
            path+= p + '/'
        return path[:-1]

    def _update_from_args(self, cfg_dict):
        args = self.args
        for var in vars(args):
            cfg_dict[var] = getattr(args, var)
        return cfg_dict

    def _initialize_cfg(self):
        if self.need_initialization:
            self.need_initialization = False
            if os.path.exists('./configs/base.yaml'):
                with open("./configs/base.yaml", 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
            else:
                with open(os.path.realpath(__file__).split('/')[-3] + "/configs/base.yaml", 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg
    
    def _load_yaml(self, args, file_name=""):
        assert args.cfg_file is not None
        if not file_name == "": # reading from base file
            with open(file_name, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        else: 
            if os.getcwd().split("/")[-1] == args.cfg_file.split("/")[0]:
                args.cfg_file = args.cfg_file.replace(os.getcwd().split("/")[-1], "./")
            with open(args.cfg_file, 'r') as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
                    file_name = args.cfg_file

        if "_BASE_RUN" not in cfg.keys() and "_BASE_MODEL" not in cfg.keys() and "_BASE" not in cfg.keys():
            # return cfg if the base file is being accessed
            # cfg = self._merge_cfg_from_command_update(args, cfg)
            return cfg
        
        if "_BASE" in cfg.keys():
            if cfg["_BASE"][1] == '.':
                prev_count = cfg["_BASE"].count('..')
                cfg_base_file = self._path_join(file_name.split('/')[:(-1-cfg["_BASE"].count('..'))] + cfg["_BASE"].split('/')[prev_count:])
            else:
                cfg_base_file = cfg["_BASE"].replace(
                    "./", 
                    args.cfg_file.replace(args.cfg_file.split('/')[-1], "")
                )
            cfg_base = self._load_yaml(args, cfg_base_file)
            cfg = self._merge_cfg_from_base(cfg_base, cfg)
        else:
            if "_BASE_RUN" in cfg.keys():
                if cfg["_BASE_RUN"][1] == '.':
                    prev_count = cfg["_BASE_RUN"].count('..')
                    cfg_base_file = self._path_join(file_name.split('/')[:(-1-prev_count)] + cfg["_BASE_RUN"].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg["_BASE_RUN"].replace(
                        "./", 
                        args.cfg_file.replace(args.cfg_file.split('/')[-1], "")
                    )
                cfg_base = self._load_yaml(args, cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg, preserve_base=True)
            if "_BASE_MODEL" in cfg.keys():
                if cfg["_BASE_MODEL"][1] == '.':
                    prev_count = cfg["_BASE_MODEL"].count('..')
                    cfg_base_file = self._path_join(file_name.split('/')[:(-1-cfg["_BASE_MODEL"].count('..'))] + cfg["_BASE_MODEL"].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg["_BASE_MODEL"].replace(
                        "./", 
                        args.cfg_file.replace(args.cfg_file.split('/')[-1], "")
                    )
                cfg_base = self._load_yaml(args, cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg)
        cfg = self._merge_cfg_from_command(args, cfg)
        return cfg
    
    def _merge_cfg_from_base(self, cfg_base, cfg_new, preserve_base=False):
        for k,v in cfg_new.items():
            if k in cfg_base.keys():
                if isinstance(v, dict):
                    self._merge_cfg_from_base(cfg_base[k], v)
                else:
                    cfg_base[k] = v
            else:
                if "BASE" not in k or preserve_base:
                    cfg_base[k] = v
        return cfg_base

    def _merge_cfg_from_command_update(self, args, cfg):
        if len(args.opts) == 0:
            return cfg
        
        assert len(args.opts) % 2 == 0, 'Override list {} has odd length: {}.'.format(
            args.opts, len(args.opts)
        )
        keys = args.opts[0::2]
        vals = args.opts[1::2]

        for key, val in zip(keys, vals):
           cfg[key] = val

        return cfg

    def _merge_cfg_from_command(self, args, cfg):
        assert len(args.opts) % 2 == 0, 'Override list {} has odd length: {}.'.format(
            args.opts, len(args.opts)
        )
        keys = args.opts[0::2]
        vals = args.opts[1::2]

        # maximum supported depth 3
        for idx, key in enumerate(keys):
            key_split = key.split('.')
            assert len(key_split) <= 4, 'Key depth error. \nMaximum depth: 3\n Get depth: {}'.format(
                len(key_split)
            )
            assert key_split[0] in cfg.keys(), 'Non-existant key: {}.'.format(
                key_split[0]
            )
            if len(key_split) == 2:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
            elif len(key_split) == 3:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[2] in cfg[key_split[0]][key_split[1]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
            elif len(key_split) == 4:
                assert key_split[1] in cfg[key_split[0]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[2] in cfg[key_split[0]][key_split[1]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
                assert key_split[3] in cfg[key_split[0]][key_split[1]][key_split[2]].keys(), 'Non-existant key: {}.'.format(
                    key
                )
            if len(key_split) == 1:
                cfg[key_split[0]] = vals[idx]
            elif len(key_split) == 2:
                cfg[key_split[0]][key_split[1]] = vals[idx]
            elif len(key_split) == 3:
                cfg[key_split[0]][key_split[1]][key_split[2]] = vals[idx]
            elif len(key_split) == 4:
                cfg[key_split[0]][key_split[1]][key_split[2]][key_split[3]] = vals[idx]
        return cfg
    
    def _update_dict(self, config_file, cfg_dict):
        def recur(key, elem):
            if type(elem) is dict:
                return key, Config(config_file, load=False, cfg_dict=elem, cfg_level=key)
            else:
                if type(elem) is str and elem[1:3]=="e-":
                    elem = float(elem)
                return key, elem
        dic = dict(recur(k, v) for k, v in cfg_dict.items())
        self.__dict__.update(dic)
    
    def get_args(self):
        return self.args
    
    def __repr__(self):
        return "{}\n".format(self.dump())
            
    def dump(self):
        return json.dumps(self.cfg_dict, indent=2)

    def deep_copy(self):
        return copy.deepcopy(self)
    
# if __name__ == '__main__':
#     # debug
#     cfg = Config(load=True)
#     print(cfg.DATA)
