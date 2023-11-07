import importlib

def load_datamodule(config, tokenizer, batch_size, val_batch_size, train_num, val_num):
    module_name, class_name = config.datamodule.class_name.rsplit(".",maxsplit=2)
    module = importlib.import_module("."+module_name, package="datas")
    class_obj = getattr(module, class_name)
    args = config.datamodule.get("args", {})
    data_moduel = class_obj(tokenizer = tokenizer, 
                            batch_size=batch_size, val_batch_size=val_batch_size, 
                            train_num=train_num, val_num=val_num, 
                            **args)
    return data_moduel