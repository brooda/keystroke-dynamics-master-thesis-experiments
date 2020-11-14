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



    file_names =["swipe_pos_correct", "swipe_pos_frag_correct", "swipe_vel_correct", "swipe_vel_frag_correct", "swipe_acc_correct", "swipe_acc_frag_correct"]
    file_names =[f"v2_{el}" for el in file_names]

    data_types = [
        [
            (DTW_DISTANCES, [swipexpositions_signal]),
            (DTW_DISTANCES, [swipeypositions_signal])
        ],
        [
            (DTW_DISTANCES_PARTS, [swipexpositions_signalpieces]),
            (DTW_DISTANCES_PARTS, [swipeypositions_signalpieces])
        ],
        [
            (DTW_DISTANCES, [swipexpositionsvel_signal]),
            (DTW_DISTANCES, [swipeypositionsvel_signal])
        ],
        [
            (DTW_DISTANCES_PARTS, [swipexpositionsvel_signalpieces]),
            (DTW_DISTANCES_PARTS, [swipeypositionsvel_signalpieces])
        ],
        [
            (DTW_DISTANCES, [swipexpositionsacc_signal]),
            (DTW_DISTANCES, [swipeypositionsacc_signal])
        ],
        [
            (DTW_DISTANCES_PARTS, [swipexpositionsacc_signalpieces]),
            (DTW_DISTANCES_PARTS, [swipeypositionsacc_signalpieces])
        ],
    ]


    for file_name, data_type in zip(file_names, data_types):
        args["used_data"] = data_type

        process = Process(Arguments(args))
        experiment_results = process.run()

        with open(f"../ExperimentResults/{file_name}.p", 'wb') as handle:
            pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        