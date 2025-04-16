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
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mxpool = nn.MaxPool1d(1024)
        self.pd_rot = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )
        self.pd_trans = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        # raise NotImplementedError("You need to implement some modules here")

    alpha = 0.3
    beta = 1
    gamma = 0.1

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
        I = torch.eye(3).to(pred_rot.device)
        orthogonality_loss = torch.mean(torch.norm(torch.bmm(pred_rot, pred_rot.transpose(2,1)) - I, dim=(1,2)))
        return trans_loss, rot_loss, orthogonality_loss

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
        x = pc.transpose(1, 2)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mxpool(x)
        x = x.view(x.size(0), -1)
        pred_trans = self.pd_trans(x)
        pred_rot = self.pd_rot(x)
        pred_rot = pred_rot.view(-1, 3, 3)

        trans_loss, rot_loss, orthogonality_loss = self.get_loss(trans, rot, pred_trans, pred_rot)
        loss = self.alpha * trans_loss + self.beta * rot_loss + self.gamma * orthogonality_loss
    
        metric = dict(
            loss=loss,
            trans_loss = trans_loss, 
            rot_loss = rot_loss,
            orthogonality_loss = orthogonality_loss
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
        x = pc.transpose(1, 2)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mxpool(x)
        x = x.view(x.size(0), -1)
        pred_trans = self.pd_trans(x)
        pred_rot = self.pd_rot(x)
        pred_rot = pred_rot.view(-1, 3, 3)
        pred_rot_ortho = self.svd_orthogonalization(pred_rot[:, ])

        return pred_trans, pred_rot_ortho
        # raise NotImplementedError("You need to implement the est function")

    