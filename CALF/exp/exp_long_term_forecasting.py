from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.cmLoss import cmLoss

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, vali_test=False):
        data_set, data_loader = data_provider(self.args, flag, vali_test)
        return data_set, data_loader

    def _select_optimizer(self):
        param_dict = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and "_proj" in n], "lr": 1e-4},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and "_proj" not in n], "lr": self.args.learning_rate},
        ]
        model_optim = optim.Adam([param_dict[1]], lr=self.args.learning_rate)
        loss_optim = optim.Adam([param_dict[0]], lr=self.args.learning_rate)
        return model_optim, loss_optim

    def _select_criterion(self):
        criterion = cmLoss(
            self.args.feature_loss,
            self.args.output_loss,
            self.args.task_loss,
            self.args.task_name,
            self.args.feature_w,
            self.args.output_w,
            self.args.task_w,
        )
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test", vali_test=True)

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim, loss_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                loss_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs_dict = self.model(batch_x)
                loss = criterion(outputs_dict, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                loss_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            if self.args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]["lr"]))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                outputs_ensemble = outputs["outputs_time"][:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :]

                pred = outputs_ensemble.detach().cpu()
                true = batch_y.detach().cpu()
                loss = F.mse_loss(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        if self.args.zero_shot:
            self.args.data = self.args.target_data
            self.args.data_path = f"{self.args.data}.csv"

        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            model_path = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join("./checkpoints", setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        preds = []
        trues = []
        inputs = []

        folder_path = os.path.join("./test_results", setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x[:, -self.args.seq_len :, :])
                outputs_ensemble = outputs["outputs_time"][:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :]

                pred = outputs_ensemble.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                hist = batch_x.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                inputs.append(hist)

                if i % 20 == 0:
                    plot_input = hist.copy()
                    plot_true = true.copy()
                    plot_pred = pred.copy()
                    if test_data.scale and self.args.inverse:
                        shape = plot_input.shape
                        plot_input = test_data.inverse_transform(plot_input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((plot_input[0, :, -1], plot_true[0, :, -1]), axis=0)
                    pd = np.concatenate((plot_input[0, :, -1], plot_pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.array(preds)
        trues = np.array(trues)
        inputs = np.array(inputs)
        print("test shape:", preds.shape, trues.shape, inputs.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        print("test shape:", preds.shape, trues.shape, inputs.shape)

        folder_path = os.path.join("./results", setting)
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))

        with open("result_long_term_forecast.txt", "a") as f:
            f.write(setting + " \n")
            f.write("mse:{}, mae:{}".format(mse, mae))
            f.write("\n\n")

        np.save(os.path.join(folder_path, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, "pred.npy"), preds)
        np.save(os.path.join(folder_path, "true.npy"), trues)
        np.save(os.path.join(folder_path, "input.npy"), inputs)
        return
