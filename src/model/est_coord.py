from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from ..config import Config
from ..vis import Vis

class EstCoordNet(nn.Module):

    config: Config
    eps = 0.1
    iters = 100

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mxpool = nn.MaxPool1d(1024)
        self.mlp4 = nn.Sequential(
            nn.Conv1d(1088, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.mlp5 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.mlp6 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv = nn.Conv1d(128, 3, kernel_size=3, padding=1)

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # raise NotImplementedError("You need to implement the forward function")
        pc = pc.transpose(1, 2)
        x = self.mlp1(pc)
        y = self.mlp2(x)
        y = self.mlp3(y)
        y = self.mxpool(y)
        y = y.view(y.size(0), -1)
        y = y.unsqueeze(2)
        y = y.expand(x.size(0), -1, x.size(2))
        x = torch.cat((x, y), dim=1)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.mlp6(x)
        x = self.conv(x)
        pred_coord = x.transpose(1, 2)
        loss = nn.MSELoss()(pred_coord, coord)
        metric = dict(
            loss=loss,
            # additional metrics you want to log
        )
        return loss, metric

    def calc_param(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centroid_q = torch.mean(q, dim=1, keepdim=True)
        centroid_p = torch.mean(p, dim=1, keepdim=True)
        q_centered = q - centroid_q  # BxNx3
        p_centered = p - centroid_p  # BxNx3
        H = torch.bmm(q_centered.transpose(1, 2), p_centered)
        U, S, Vh = torch.svd(H)
        R = torch.bmm(Vh, U.transpose(1, 2))  # Bx3x3
        det_R = torch.det(R)  # B
        Vh_corrected = Vh.clone()
        Vh_corrected[det_R < 0, :, -1] *= -1  # 对行列式<0的Batch，反转最后一列
        R = torch.bmm(Vh_corrected, U.transpose(1, 2))  # Bx3x3
        centroid_q = centroid_q.squeeze(1)  # Bx3
        t = centroid_p.squeeze(1) - torch.bmm(R, centroid_q.unsqueeze(-1)).squeeze(-1)  # Bx3
        return t, R

    def RANSAC(self, p: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p.transpose_(1, 2)
        q.transpose_(1, 2)
        B, N, _ = p.shape
        sample_indices = torch.randint(N, (B, self.iters, 3), device=p.device)
        q_sampled = q[torch.arange(B)[:, None, None], sample_indices, :]  # (B, iters, 3, 3)
        p_sampled = p[torch.arange(B)[:, None, None], sample_indices, :]
        p_sampled = p_sampled.view(B*self.iters, 3, 3)
        q_sampled = q_sampled.view(B*self.iters, 3, 3)
        t, R = self.calc_param(p_sampled, q_sampled)
        # print(t.shape, R.shape)
        # t = t.view(B, self.iters, 3)
        # R = R.view(B, self.iters, 3, 3)
        q_tmp = q.view(B, 1, N, 3).expand(-1, self.iters, -1, -1)  # BxitersxNx3
        p_tmp = p.view(B, 1, N, 3).expand(-1, self.iters, -1, -1)  # BxitersxNx3
        p_tmp = p_tmp.reshape(B*self.iters, N, 3)
        q_tmp = q_tmp.reshape(B*self.iters, N, 3)
        err = torch.nn.functional.mse_loss(q_tmp, torch.bmm(p_tmp, R.transpose(1, 2)) + t.unsqueeze(1), reduction='none').sum(dim=-1)  # BxitersxN
        inlies = err <= self.eps # B*iters*N
        inlies_count = inlies.sum(dim=-1, keepdim=False) 
        inlies_count = inlies_count.view(B, self.iters)
        idx = torch.argmax(inlies_count, dim=1)
        inlies = inlies.view(B, self.iters, N)
        # best_inlier_mask = inlies[torch.arange(B), idx]
        # print(best_inlier_mask.shape)
        # p_inlies_points = p[torch.arange(B), inlies[torch.arange(B), idx]]  # BxNx3
        # q_inlies_points = q[torch.arange(B), inlies[torch.arange(B), idx]]  # BxNx3
        batch_indices = torch.arange(B)
        mask = inlies[batch_indices, idx, :] # (B, N)
        mask = mask.unsqueeze(-1) # (B, N, 1)
        p_inlies_points = p * mask
        q_inlies_points = q * mask
        # print(inlies[idx].shape)
        # print(p.shape)
        # p_inlies_points = p[best_inlier_mask]
        # q_inlies_points = q[best_inlier_mask]
        # print(p_inlies_points.shape, q_inlies_points.shape)
        return self.calc_param(p_inlies_points, q_inlies_points)

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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        
        pc = pc.transpose(1, 2)
        x = self.mlp1(pc)
        y = self.mlp2(x)
        y = self.mlp3(y)
        y = self.mxpool(y)
        y = y.view(y.size(0), -1)
        y = y.unsqueeze(2)
        y = y.expand(x.size(0), -1, x.size(2))
        x = torch.cat((x, y), dim=1)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.mlp6(x)
        x = self.conv(x)
        return self.RANSAC(x, pc)

        # raise NotImplementedError("You need to implement the est function")
