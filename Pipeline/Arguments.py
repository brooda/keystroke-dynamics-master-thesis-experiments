class Arguments:
    def __init__(self, args):
        self.data_path = args["data_path"]
        self.train_sessions = args["train_sessions"]
        self.test_sessions = args["test_sessions"]
        self.repetitions = args["repetitions"]
        self.used_data = args["used_data"]
        self.models = args["models"]
