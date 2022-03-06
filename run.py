from fire import Fire
from addict import Dict
import yaml
from train import train
import pandas as pd

def update_config(config, params):

    for k, v in params.items():
        *path, key = k.split(".")
        config.update({k: v})
        print(f"Overwriting {k} = {v} (was {config.get(key)})")
    return config


def fit(**kwargs):

    # use base config
    with open("./configs/us/base.yml") as cfg:
        base_config = yaml.load(cfg, Loader=yaml.FullLoader)

    if "config" in kwargs.keys():
        cfg_path = kwargs["config"]

        with open(cfg_path) as cfg:
            cfg_yaml = yaml.load(cfg, Loader=yaml.FullLoader)

        merged_cfg = update_config(base_config, cfg_yaml)
    else:
        merged_cfg = base_config

    update_cfg = update_config(merged_cfg, kwargs)
    return update_cfg

def main(opt):
    print(opt)
    df = pd.read_csv(opt.df_path, index_col=0)

    if opt.kfold:
        for f in df['fold'].unique():
            train(df, f, opt)
    else:
        # use 0 fold as validation
        train(df, 0, opt)

if __name__ == '__main__':
    cfg = Dict(Fire(fit))
    main(cfg)
