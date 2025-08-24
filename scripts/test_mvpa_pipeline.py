from analysis.mvpa.loaders import ExperimentDataLoader

def main():
    loader = ExperimentDataLoader(data_dir='/project/3018040.05/bids/')
    breakpoint()
    train_dataset, test_dataset = loader.load_experiment_1_data(subject_id='sub-001')
    

if __name__=="__main__":
    main()