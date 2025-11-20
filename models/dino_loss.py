import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        # Running center for teacher logits
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor):
 
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

    @torch.no_grad()
    def teacher_distribution(self, logits: torch.Tensor):

        centered = (logits - self.center) / self.teacher_temp
        return F.softmax(centered, dim=-1)

    def forward(self, student_logits_list, teacher_logits_list):

        # Update center using teacher global outputs
        all_teacher = torch.cat(teacher_logits_list, dim=0)
        self.update_center(all_teacher)

        total_loss = 0.0
        n_loss_terms = 0

        # Compute teacher soft targets
        teacher_probs = [self.teacher_distribution(t) for t in teacher_logits_list]

        # Compute cross-view loss
        for s in student_logits_list:
            student_logp = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_probs:
                loss = -(t * student_logp).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        return total_loss / n_loss_terms
