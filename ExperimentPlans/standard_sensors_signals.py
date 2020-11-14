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



    file_names =["standard_accx_signal",
        "standard_accx_signalpieces", \
        "standard_accy_signal", "standard_accy_signalpieces", \
        "standard_accz_signal", "standard_accz_signalpieces", \
        "standard_rotx_signal", "standard_rotx_signalpieces", \
        "standard_roty_signal", "standard_roty_signalpieces", \
        "standard_rotz_signal", "standard_rotz_signalpieces" ]

    file_names =[f"v2_{el}" for el in file_names]

    data_types = [
        [
            (DTW_DISTANCES, [accel_x_signal])
        ],
        [
            (DTW_DISTANCES_PARTS, [accel_x_signalpieces])
        ],
        [
            (DTW_DISTANCES, [accel_y_signal])
        ],
        [
            (DTW_DISTANCES_PARTS, [accel_y_signalpieces])
        ],
        [
            (DTW_DISTANCES, [accel_z_signal])
        ],
        [
            (DTW_DISTANCES_PARTS, [accel_z_signalpieces])
        ],
        [
            (DTW_DISTANCES, [rotation_alpha_signal])
        ],
        [
            (DTW_DISTANCES_PARTS, [rotation_alpha_signalpieces])
        ],
        [
            (DTW_DISTANCES, [rotation_beta_signal])
        ],
        [
            (DTW_DISTANCES_PARTS, [rotation_beta_signalpieces])
        ],
        [
            (DTW_DISTANCES, [rotation_gamma_signal])
        ],
        [
            (DTW_DISTANCES_PARTS, [rotation_gamma_signalpieces])
        ]
    ]

    for file_name, data_type in zip(file_names, data_types):
        print("data type: ", data_type)
        args["used_data"] = data_type

        process = Process(Arguments(args))
        experiment_results = process.run()

        with open(f"../ExperimentResults/{file_name}.p", 'wb') as handle:
            pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)