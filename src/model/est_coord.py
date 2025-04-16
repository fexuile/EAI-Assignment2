from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from ..config import Config
from ..vis import Vis

class EstCoordNet(nn.Module):

    config: Config
    ransac_eps = 0.1
    ransac_iterations = 100

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
        q = q.transpose(1, 2)
        p = p.transpose(1, 2)
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

    def svd_solve(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_cent = q.mean(dim=1, keepdim=True)
        p_cent = p.mean(dim=1, keepdim=True)
        q_centered = q - q_cent
        p_centered = p - p_cent
        
        H = q_centered @ p_centered.T
        try:
            U, S, Vt = torch.linalg.svd(H)
        except:
            return torch.eye(3, device=q.device, dtype=q.dtype), torch.zeros(3, 1, device=q.device, dtype=q.dtype)
        
        R = Vt.T @ U.T
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = p_cent - R @ q_cent
        return R, t

    def ransac_single(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = q.size(1)
        min_samples = 3
        best_num_inliers = 0
        best_R = torch.eye(3, device=q.device, dtype=q.dtype)
        best_t = torch.zeros(3, 1, device=q.device, dtype=q.dtype)
        
        if n < min_samples:
            return self.svd_solve(q, p)
        
        for _ in range(self.ransac_iterations):
            sample_idx = torch.randperm(n, device=q.device)[:min_samples]
            q_samples = q[:, sample_idx]
            p_samples = p[:, sample_idx]
            
            R, t = self.svd_solve(q_samples, p_samples)
            if R is None:
                continue
            
            transformed_q = R @ q + t
            errors = torch.sum((transformed_q - p) ** 2, dim=0)
            num_inliers = (errors < self.ransac_eps ** 2).sum().item()
            
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_R = R.clone()
                best_t = t.clone()
        
        if best_num_inliers == 0:
            return self.svd_solve(q, p)
        else:
            transformed_q = best_R @ q + best_t
            errors = torch.sum((transformed_q - p) ** 2, dim=0)
            inlier_mask = errors < self.ransac_eps ** 2
            if inlier_mask.sum() >= min_samples:
                R_refined, t_refined = self.svd_solve(q[:, inlier_mask], p[:, inlier_mask])
                if R_refined is not None:
                    return R_refined, t_refined
        return best_R, best_t

    def RANSAC(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)
        device = q.device
        dtype = q.dtype
        trans = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        rot = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        for i in range(batch_size):
            q_i = q[i]  # (3, n)
            p_i = p[i]
            R, t = self.ransac_single(q_i, p_i)
            rot[i] = R
            trans[i] = t.view(3)

        return trans, rot

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
