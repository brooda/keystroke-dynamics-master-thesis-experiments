if __name__ == '__main__': 
    import sys
    sys.path.append('..')
    from Pipeline.Process import Process
    from Pipeline.Constants import *
    import pickle
    from Pipeline.Arguments import Arguments

    args = {}
    args["data_path"] = "../data/computer_maxion/data_transformed.csv"
    
    args["train_sessions"] =  [0, 1]
    args["test_sessions"] = [2]

    args["repetitions"] = (0, 14)

    args["used_data"] = [
        (DIFF_TRANSFORMATION, [down_up]),
    ]
    args["models"] = [KNN, RANDOM_FOREST, SVC, NEURAL_NETWORK]

    print("small set")
    process = Process(Arguments(args))
    experiment_results = process.run()

    with open("../ExperimentResults/computer_maxion.p", 'wb') as handle:
        pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("big set")
    args["train_sessions"] =  [0, 1, 2, 3, 4]
    args["test_sessions"] = [5, 6]

    process = Process(Arguments(args))
    experiment_results = process.run()

    with open("../ExperimentResults/computer_maxion_bigger.p", 'wb') as handle:
        pickle.dump(experiment_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
