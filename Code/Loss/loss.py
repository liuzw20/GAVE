import torch.nn as nn
import torch
import torchvision.utils as vutils


class BCE3Loss(nn.Module):  
    """ 
        A = 0+1, Vessel = 0+1+2, V = 1+2
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_vessels, vessels, mask):
        mask = mask[:, 0, :, :]
        mask = torch.round(mask)

        pred_a = pred_vessels[:, 0, :, :]
        pred_vt = pred_vessels[:, 1, :, :]
        pred_v = pred_vessels[:, 2, :, :]
        
        gt_a = vessels[:, 0, :, :] + vessels[:, 1, :, :]              # A = 0 + 1
        gt_vt = vessels[:, 0, :, :] + vessels[:, 1, :, :] + vessels[:, 2, :, :]  # Vessel tree = 0 + 1 + 2
        gt_v = vessels[:, 1, :, :] + vessels[:, 2, :, :]              # V = 1 + 2

        gt_a = torch.clamp(gt_a, 0, 1)
        gt_vt = torch.clamp(gt_vt, 0, 1)
        gt_v = torch.clamp(gt_v, 0, 1)

        loss = self.loss(pred_a[mask > 0.], gt_a[mask > 0.])
        loss += self.loss(pred_vt[mask > 0.], gt_vt[mask > 0.])
        loss += self.loss(pred_v[mask > 0.], gt_v[mask > 0.])
        
        return loss

    def save_predicted(self, prediction, fname):
        prediction_processed = self.process_predicted(prediction)
        vutils.save_image(prediction_processed, fname)

    def process_predicted(self, prediction):
        return torch.sigmoid(prediction.clone())


class RRLoss(nn.Module):
    """
    Recursive refinement loss.
    """
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, predictions, gt, mask):
        loss_1 = self.base_criterion(predictions[0], gt, mask)
        if len(predictions) == 1:
            return loss_1

        # Second loss (refinement) inspired by Mosinska:CVPR:2018.
        loss_2 = 1 * self.base_criterion(predictions[1], gt, mask)
        if len(predictions) == 2:
            return loss_1 + loss_2
        for i, prediction in enumerate(predictions[2:], 2):
            loss_2 += i * self.base_criterion(prediction, gt, mask)

        K = len(predictions[1:])
        Z = (1/2) * K * (K + 1)

        loss_2 *= 1/Z

        loss = loss_1 + loss_2

        return loss

    def save_predicted(self, predictions, fname):
        self.base_criterion.save_predicted(predictions[-1], fname)

    def process_predicted(self, predictions):
        new_predictions = []
        for prediction in predictions:
            new_predictions.append(self.base_criterion.process_predicted(prediction))
        return new_predictions
