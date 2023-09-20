import importlib

def load_datamodule(config, tokenizer, batch_size, val_batch_size):
    module_name, class_name = config.datamodule.class_name.rsplit(".",maxsplit=2)
    module = importlib.import_module("."+module_name, package="datas")
    class_obj = getattr(module, class_name)
    if "args" in config.datamodule:
        data_moduel = class_obj(tokenizer = tokenizer, batch_size=batch_size, val_batch_size=val_batch_size,  **config.datamodule.args)
    else:
        data_moduel = class_obj(tokenizer = tokenizer, batch_size=batch_size, val_batch_size=val_batch_size)
    return data_moduel