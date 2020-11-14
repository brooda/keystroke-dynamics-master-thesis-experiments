if __name__ == '__main__': 
    import sys
    sys.path.append('..')
    from Pipeline.Process import Process
    from Pipeline.Constants import *
    import pickle
    from Pipeline.Arguments import Arguments

    args = {}
    args["data_path"] = "../data/mobile_my/data_transformed.csv"
    
    args["train_sessions"] =  [0, 1]
    args["test_sessions"] = [2]

    args["repetitions"] = (2, 16)

    args["models"] = [KNN, RANDOM_FOREST, SVC, NEURAL_NETWORK]

    file_names =["mobile_dup", "mobile_dup_xy", "mobile_dup_xytoeplitz", \
        "mobile_accx", "mobile_accy", "mobile_accz", \
        "mobile_accxy","mobile_accxz", "mobile_accyz", "mobile_accxyz"]
    file_names =[f"v2_{el}" for el in file_names]

    data_types = [
        [
            (DIFF_TRANSFORMATION, [down_up])
        ],
        [
            (DIFF_TRANSFORMATION, [down_up]),
            (NO_TRANSFORMATION, [normalized_x]),
            (NO_TRANSFORMATION, [normalized_y])
        ],
        [
            (DIFF_TRANSFORMATION, [down_up]),
            (TOEPLITZ_TRANSFORMATION, [normalized_x, normalized_y])
        ],
        [
            (NO_TRANSFORMATION, [accel_x_mean]),
        ],
        [
            (NO_TRANSFORMATION, [accel_y_mean]),
        ],
        [
            (NO_TRANSFORMATION, [accel_z_mean]),
        ],
        [
            (NO_TRANSFORMATION, [accel_x_mean]),
            (NO_TRANSFORMATION, [accel_y_mean]),
        ],
        [
            (NO_TRANSFORMATION, [accel_x_mean]),
            (NO_TRANSFORMATION, [accel_z_mean]),
        ],        
        [
            (NO_TRANSFORMATION, [accel_y_mean]),
            (NO_TRANSFORMATION, [accel_z_mean]),
        ],
        [
            (NO_TRANSFORMATION, [accel_x_mean]),
            (NO_TRANSFORMATION, [accel_y_mean]),
            (NO_TRANSFORMATION, [accel_z_mean]),
        ],
    ]


    for file_name, data_type in zip(file_names, data_types):
        print("data type: ", data_type)
        args["used_data"] = data_type

        process = Process(Arguments(args))
        experiment_results = process.run()

        with open(f"../ExperimentResults/{file_name}.p", 'wb') as handle:
            pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)