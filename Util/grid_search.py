import random
from Util.data_import import *
from Util.data_preprocessing import *
from Util.train import *
from Util.test import *

random.seed(0)

def grid_search():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_dir = 'best_model_with_mixup'                                        # choose dir to save current best model
    download_data()                                                                 # if you need to create new dataset you need to download wav file
    data = DATA('../data', wav_to_pkl=False, with_mixup=True)        # to create new pkl files choose wav_to_pkl=True, for loading data with mixup choose with_mixup=True

    for trail in range(10):
        # random model params
        num_epochs = random.randint(150, 150)
        lr = 10 ** random.uniform(-5, -3)
        batch_size = 2 ** random.randint(5, 8)
        dropout = random.uniform(0, 0.2)

        # train
        main_train_loop(data.X_train, data.y_train, data.X_val, data.y_val, num_epochs, lr, batch_size, dropout, device, best_model_dir)

    # test model
    evaluate_model_on_test(data.X_test, data.y_test,'../' + best_model_dir, device)
    evaluate_model_on_test(data.X_test_with_mixup, data.y_test_with_mixup, '../' + best_model_dir, device)
    # To view the training progress for all models in a given directory, enter the following command in the PyCharm terminal:
    #       tensorboard --logdir=<Enter your result dir>
    # example:
    #       tensorboard --logdir=runs
if __name__ == '__main__':
    grid_search()