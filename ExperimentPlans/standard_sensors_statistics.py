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



    file_names =["standard_accx_mean", "standard_accx_rms", "standard_accx_both",\
        "standard_accy_mean", "standard_accy_rms", "standard_accy_both",\
        "standard_accz_mean", "standard_accz_rms", "standard_accz_both",\
        "standard_rotx_mean", "standard_rotx_rms", "standard_rotx_both",\
        "standard_roty_mean", "standard_roty_rms", "standard_roty_both",\
        "standard_rotz_mean", "standard_rotz_rms", "standard_rotz_both"\
    ]

    file_names =[f"v2_{el}" for el in file_names]

    data_types = [
        [
            (NO_TRANSFORMATION, [accel_x_mean])
        ],
        [
            (NO_TRANSFORMATION, [accel_x_rms])
        ],
        [
            (NO_TRANSFORMATION, [accel_x_mean]),
            (NO_TRANSFORMATION, [accel_x_rms])
        ],
        [
            (NO_TRANSFORMATION, [accel_y_mean])
        ],
        [
            (NO_TRANSFORMATION, [accel_y_rms])
        ],
        [
            (NO_TRANSFORMATION, [accel_y_mean]),
            (NO_TRANSFORMATION, [accel_y_rms])
        ],
        [
            (NO_TRANSFORMATION, [accel_z_mean])
        ],
        [
            (NO_TRANSFORMATION, [accel_z_rms])
        ],
        [
            (NO_TRANSFORMATION, [accel_z_mean]),
            (NO_TRANSFORMATION, [accel_z_rms])
        ], 
        [
            (NO_TRANSFORMATION, [rotation_alpha_mean])
        ],
        [
            (NO_TRANSFORMATION, [rotation_alpha_rms])
        ],
        [
            (NO_TRANSFORMATION, [rotation_alpha_mean]),
            (NO_TRANSFORMATION, [rotation_alpha_rms])
        ], 
        [
            (NO_TRANSFORMATION, [rotation_beta_mean])
        ],
        [
            (NO_TRANSFORMATION, [rotation_beta_rms])
        ],
        [
            (NO_TRANSFORMATION, [rotation_beta_mean]),
            (NO_TRANSFORMATION, [rotation_beta_rms])
        ], 
        [
            (NO_TRANSFORMATION, [rotation_gamma_mean])
        ],
        [
            (NO_TRANSFORMATION, [rotation_gamma_rms])
        ],
        [
            (NO_TRANSFORMATION, [rotation_gamma_mean]),
            (NO_TRANSFORMATION, [rotation_gamma_rms])
        ], 
    ]

    for file_name, data_type in zip(file_names, data_types):
        print("data type: ", data_type)
        args["used_data"] = data_type

        process = Process(Arguments(args))
        experiment_results = process.run()

        with open(f"../ExperimentResults/{file_name}.p", 'wb') as handle:
            pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)