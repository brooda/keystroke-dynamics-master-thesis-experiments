if __name__ == '__main__': 
    import sys
    sys.path.append('..')
    from Pipeline.Process import Process
    from Pipeline.Constants import *
    import pickle
    from Pipeline.Arguments import Arguments

    args = {}
    args["data_path"] = "../data/swipe/data_transformed.csv"
    
    args["train_sessions"] =  [0, 1]
    args["test_sessions"] = [2]

    args["repetitions"] =(2, 16)

    args["models"] = [KNN, RANDOM_FOREST, SVC, NEURAL_NETWORK]



    file_names =["swipe_stat_pos_mean", "swipe_stat_pos_rms", "swipe_stat_pos_both", \
        "swipe_stat_vel_mean", "swipe_stat_vel_rms", "swipe_stat_vel_both", \
        "swipe_stat_acc_mean", "swipe_stat_acc_rms", "swipe_stat_acc_both"]
    
    file_names =[f"v2_{el}" for el in file_names]

    data_types = [
        [
            (NO_TRANSFORMATION, [swipexpositions_mean]),
            (NO_TRANSFORMATION, [swipeypositions_mean])
        ],
        [
            (NO_TRANSFORMATION, [swipexpositions_rms]),
            (NO_TRANSFORMATION, [swipeypositions_rms])
        ],        
        [
            (NO_TRANSFORMATION, [swipexpositions_mean]),
            (NO_TRANSFORMATION, [swipeypositions_mean]),
            (NO_TRANSFORMATION, [swipeypositions_rms]),
            (NO_TRANSFORMATION, [swipeypositions_rms])
        ],
        [
            (NO_TRANSFORMATION, [swipexpositionsvel_mean]),
            (NO_TRANSFORMATION, [swipeypositionsvel_mean])
        ],
        [
            (NO_TRANSFORMATION, [swipexpositionsvel_rms]),
            (NO_TRANSFORMATION, [swipeypositionsvel_rms])
        ],        
        [
            (NO_TRANSFORMATION, [swipexpositionsvel_mean]),
            (NO_TRANSFORMATION, [swipeypositionsvel_mean]),
            (NO_TRANSFORMATION, [swipeypositionsvel_rms]),
            (NO_TRANSFORMATION, [swipeypositionsvel_rms])
        ],
        [
            (NO_TRANSFORMATION, [swipexpositionsacc_mean]),
            (NO_TRANSFORMATION, [swipeypositionsacc_mean])
        ],
        [
            (NO_TRANSFORMATION, [swipexpositionsacc_rms]),
            (NO_TRANSFORMATION, [swipeypositionsacc_rms])
        ],        
        [
            (NO_TRANSFORMATION, [swipexpositionsacc_mean]),
            (NO_TRANSFORMATION, [swipeypositionsacc_mean]),
            (NO_TRANSFORMATION, [swipeypositionsacc_rms]),
            (NO_TRANSFORMATION, [swipeypositionsacc_rms])
        ]
    ]


    for file_name, data_type in zip(file_names, data_types):
        args["used_data"] = data_type

        process = Process(Arguments(args))
        experiment_results = process.run()

        with open(f"../ExperimentResults/{file_name}.p", 'wb') as handle:
            pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
