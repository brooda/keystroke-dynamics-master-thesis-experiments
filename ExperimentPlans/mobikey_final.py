if __name__ == '__main__': 
    import sys
    sys.path.append('..')
    from Pipeline.Process import Process
    from Pipeline.Constants import *
    import pickle
    from Pipeline.Arguments import Arguments

    args = {}
    args["data_path"] = "../data/mobile_mobikey/data_transformed.csv"
    
    args["train_sessions"] =  [0, 1]
    args["test_sessions"] = [2]

    args["repetitions"] = (2, 16)

    args["models"] = [KNN, RANDOM_FOREST, SVC, NEURAL_NETWORK]

    file_names =["mobikey_final"]
    file_names =[f"v2_{el}" for el in file_names]

    data_types = [

        [
            (DIFF_TRANSFORMATION, [down_up]),
            (NO_TRANSFORMATION, [normalized_x]),
            (NO_TRANSFORMATION, [normalized_y]),
            (NO_TRANSFORMATION, [gravity_x]),
            (NO_TRANSFORMATION, [gravity_y]),
            (NO_TRANSFORMATION, [gravity_z]),
        ]
    ]

    for file_name, data_type in zip(file_names, data_types):
        args["used_data"] = data_type

        process = Process(Arguments(args))
        experiment_results = process.run()

        with open(f"../ExperimentResults/{file_name}.p", 'wb') as handle:
            pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)