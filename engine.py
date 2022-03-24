import os
import sys
import numpy as np
import copy

from models import build_classification_model, save_checkpoint
from trainer import train_one_epoch, test_classification, evaluate
from utils import metric_AUROC

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


sys.setrecursionlimit(40000)


def classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases=None):
    device = torch.device(args.device)
    cudnn.benchmark = True

    #args.exp_name = args.model_name + "_" + args.init
    #model_path = os.path.join("./Models/Classification",args.data_set)
    #output_path = os.path.join("./Outputs/Classification",args.data_set)

    model_path = os.path.join(model_path, args.exp_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # training phase
    if args.mode == "train":
        data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=True)
        data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers, pin_memory=True)
        log_file = os.path.join(
            model_path, "models_" + str(args.exp_no) + ".log")

        # training phase
        print("start training....")
        for i in range(args.start_index, args.num_trial):
            print("run:", str(i+1))
            start_epoch = 0
            init_loss = 1000000
            experiment = args.exp_name + "_run_" + str(i + args.exp_no)
            best_val_loss = init_loss
            patience_counter = 0
            save_model_path = os.path.join(model_path, experiment)
            criterion = torch.nn.BCELoss()

            model = build_classification_model(args)
            print(model)

            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)

            parameters = list(
                filter(lambda p: p.requires_grad, model.parameters()))

            optimizer = torch.optim.Adam(parameters, lr=args.lr)
            lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience // 2, mode='min',
                                             threshold=0.0001, min_lr=0, verbose=True)

            if args.resume:
                resume = os.path.join(model_path, experiment + '.pth.tar')
                if os.path.isfile(resume):
                    print("=> loading checkpoint '{}'".format(resume))
                    checkpoint = torch.load(resume)

                    start_epoch = checkpoint['epoch']
                    init_loss = checkpoint['lossMIN']
                    model.load_state_dict(checkpoint['state_dict'])
                    lr_scheduler.load_state_dict(checkpoint['scheduler'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
                          .format(resume, start_epoch, init_loss))
                else:
                    print("=> no checkpoint found at '{}'".format(resume))

            for epoch in range(start_epoch, args.num_epoch):
                train_one_epoch(data_loader_train, device,
                                model, criterion, optimizer, epoch)

                val_loss = evaluate(data_loader_val, device, model, criterion)

                lr_scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'lossMIN': best_val_loss,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                    },  filename=save_model_path)

                    print(
                        "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss,
                                                                                                           save_model_path))
                    best_val_loss = val_loss
                    patience_counter = 0

                else:
                    print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(
                        epoch, best_val_loss))
                    patience_counter += 1

                if patience_counter > args.patience:
                    print("Early Stopping")
                    break

            # log experiment
            # experiment = args.exp_name + "_run_" + str(i + args.exp_no)
            with open(log_file, 'a') as f:
                f.write(experiment + "\n")
                f.close()

    print("start testing.....")
    output_file = os.path.join(output_path, args.exp_name + "_results.txt")

    data_loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

    log_file = os.path.join(model_path, "models_" + str(args.exp_no) + ".log")
    if not os.path.isfile(log_file):
        print("log_file ({}) not exists!".format(log_file))
    else:
        mean_auc = []
        with open(log_file, 'r') as reader, open(output_file, 'a') as writer:
            experiment = reader.readline()
            print(">> Disease = {}".format(diseases))
            writer.write("Disease = {}\n".format(diseases))

            while experiment:
                experiment = experiment.replace('\n', '')
                saved_model = os.path.join(model_path, experiment + ".pth.tar")

                y_test, p_test = test_classification(
                    saved_model, data_loader_test, device, args)

                if test_diseases is not None:
                    y_test = copy.deepcopy(y_test[:, test_diseases])
                    p_test = copy.deepcopy(p_test[:, test_diseases])
                    individual_results = metric_AUROC(
                        y_test, p_test, len(test_diseases))
                else:
                    individual_results = metric_AUROC(
                        y_test, p_test, args.num_class)
                print(">>{}: AUC = {}".format(experiment, np.array2string(
                    np.array(individual_results), precision=4, separator=',')))
                writer.write(
                    "{}: AUC = {}\n".format(experiment, np.array2string(np.array(individual_results), precision=4, separator='\t')))

                mean_over_all_classes = np.array(individual_results).mean()
                print(">>{}: AUC = {:.4f}".format(
                    experiment, mean_over_all_classes))
                writer.write("{}: AUC = {:.4f}\n".format(
                    experiment, mean_over_all_classes))

                mean_auc.append(mean_over_all_classes)
                experiment = reader.readline()

            mean_auc = np.array(mean_auc)
            print(">> All trials: mAUC  = {}".format(
                np.array2string(mean_auc, precision=4, separator=',')))
            writer.write("All trials: mAUC  = {}\n".format(
                np.array2string(mean_auc, precision=4, separator='\t')))
            print(">> Mean AUC over All trials: = {:.4f}".format(
                np.mean(mean_auc)))
            writer.write(
                "Mean AUC over All trials = {:.4f}\n".format(np.mean(mean_auc)))
            print(">> STD over All trials:  = {:.4f}".format(np.std(mean_auc)))
            writer.write(
                "STD over All trials:  = {:.4f}\n".format(np.std(mean_auc)))
