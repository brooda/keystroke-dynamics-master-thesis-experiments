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

    file_names =["standard_accxyz_all", "standard_rotxyz_all"]

    file_names =[f"v2_{el}" for el in file_names]

    data_types = [
        [
            (DIFF_TRANSFORMATION, [down_up]),
            (NO_TRANSFORMATION, [normalized_x]),
            (NO_TRANSFORMATION, [normalized_y]),
            (NO_TRANSFORMATION, [accel_x_mean]),
            (NO_TRANSFORMATION, [accel_y_mean]),
            (NO_TRANSFORMATION, [accel_z_mean]),
            (NO_TRANSFORMATION, [accel_x_rms]),
            (NO_TRANSFORMATION, [accel_y_rms]),
            (NO_TRANSFORMATION, [accel_z_rms]),
        ],
        [
            (DIFF_TRANSFORMATION, [down_up]),
            (NO_TRANSFORMATION, [normalized_x]),
            (NO_TRANSFORMATION, [normalized_y]),
            (NO_TRANSFORMATION, [rotation_alpha_mean]),
            (NO_TRANSFORMATION, [rotation_beta_mean]),
            (NO_TRANSFORMATION, [rotation_gamma_mean]),
            (NO_TRANSFORMATION, [rotation_alpha_rms]),
            (NO_TRANSFORMATION, [rotation_beta_rms]),
            (NO_TRANSFORMATION, [rotation_gamma_rms])
        ],
    ]

    '''
    file_names =[
        "standard_accxyz_mean", "standard_accxyz_rms", "standard_accxyz_both",\
        "standard_rotxyz_mean", "standard_rotxyz_rms", "standard_rotxyz_both",\
    ]

    file_names =[f"v2_{el}" for el in file_names]

    data_types = [
        [
            (NO_TRANSFORMATION, [accel_x_mean]),
            (NO_TRANSFORMATION, [accel_y_mean]),
            (NO_TRANSFORMATION, [accel_z_mean])
        ],
        [
            (NO_TRANSFORMATION, [accel_x_rms]),
            (NO_TRANSFORMATION, [accel_y_rms]),
            (NO_TRANSFORMATION, [accel_z_rms])
        ],
        [
            (NO_TRANSFORMATION, [accel_x_mean]),
            (NO_TRANSFORMATION, [accel_y_mean]),
            (NO_TRANSFORMATION, [accel_z_mean]),
            (NO_TRANSFORMATION, [accel_x_rms]),
            (NO_TRANSFORMATION, [accel_y_rms]),
            (NO_TRANSFORMATION, [accel_z_rms])
        ],
        [
            (NO_TRANSFORMATION, [rotation_alpha_mean]),
            (NO_TRANSFORMATION, [rotation_beta_mean]),
            (NO_TRANSFORMATION, [rotation_gamma_mean])
        ],
        [
            (NO_TRANSFORMATION, [rotation_alpha_rms]),
            (NO_TRANSFORMATION, [rotation_beta_rms]),
            (NO_TRANSFORMATION, [rotation_gamma_rms])
        ],
        [
            (NO_TRANSFORMATION, [rotation_alpha_mean]),
            (NO_TRANSFORMATION, [rotation_beta_mean]),
            (NO_TRANSFORMATION, [rotation_gamma_mean]),
            (NO_TRANSFORMATION, [rotation_alpha_rms]),
            (NO_TRANSFORMATION, [rotation_beta_rms]),
            (NO_TRANSFORMATION, [rotation_gamma_rms])
        ],
    ]
    '''

    for file_name, data_type in zip(file_names, data_types):
        print("data type: ", data_type)
        args["used_data"] = data_type

        process = Process(Arguments(args))
        experiment_results = process.run()

        with open(f"../ExperimentResults/{file_name}.p", 'wb') as handle:
            pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)