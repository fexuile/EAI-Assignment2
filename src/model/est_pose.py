from typing import Tuple, Dict
import torch
from torch import nn

from ..config import Config


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU()
        self.mxpool = nn.MaxPool1d(kernel_size=1024)
        self.fc1 = nn.Linear(1024, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(256, 12)
        # raise NotImplementedError("You need to implement some modules here")

    alpha = 0.3
    beta = 0.7

    def svd_orthogonalization(self, matrix):
        U, S, V = torch.svd(matrix)  # 奇异值分解
        det = torch.det(U @ V.transpose(-1, -2))  # 计算行列式
        
        # 调整符号确保行列式为1
        sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
        V_corrected = V * sign
        R_ortho = U @ V_corrected.transpose(-1, -2)
    
        return R_ortho
    
    def get_loss(self, trans: torch.Tensor, rot: torch.Tensor, pred_trans: torch.Tensor, pred_rot: torch.Tensor) -> Tuple[float, float]:
        trans_loss = torch.nn.functional.mse_loss(pred_trans, trans)
        rot_loss = torch.nn.functional.mse_loss(pred_rot, rot)
        return trans_loss, rot_loss

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # raise NotImplementedError("You need to implement the forward function")
        pc = pc.transpose(1, 2)
        x = self.conv1(pc)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.mxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        pred_trans = x[:, :3]
        pred_rot = x[:, 3:]
        pred_rot = pred_rot.view(-1, 3, 3)
        pred_rot_ortho = self.svd_orthogonalization(pred_rot[:, ])

        trans_loss, rot_loss = self.get_loss(trans, rot, pred_trans, pred_rot_ortho)
        loss = self.alpha * trans_loss + self.beta * rot_loss
    
        metric = dict(
            loss=loss,
            trans_loss = trans_loss, 
            rot_loss = rot_loss
            # additional metrics you want to log
        )
        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.
        """
        pc = pc.transpose(1, 2)
        x = self.conv1(pc)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.mxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        pred_trans = x[:, :3]
        pred_rot = x[:, 3:]
        pred_rot = pred_rot.view(-1, 3, 3)
        pred_rot_ortho = self.svd_orthogonalization(pred_rot[:, ])

        return pred_trans, pred_rot_ortho
        # raise NotImplementedError("You need to implement the est function")

