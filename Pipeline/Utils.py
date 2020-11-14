from .Constants import *

def create_path_name(args):
    if args.path_to_save_results == NO_PATH:
        used_data_descr = ""
        for data in args.used_data:
            used_data_descr += f"_{data}"

        args.path_to_save_results = f"{args.model}{used_data_descr}"