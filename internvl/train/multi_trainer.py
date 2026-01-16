import torch
from torch.nn import functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother


class KDTrainer(Trainer):
    def __init__(self, teacher_model, mean_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.mean_prob = mean_prob
        self.label_smoother = LabelSmoother(epsilon=0.1)

    def cakld_loss(self, labels, student_logits, teacher_logits, beta_prob):
        mask = (labels != -100)

        # reverse
        teacher_output_log_prob = F.log_softmax(teacher_logits, dim=2)
        # Compute the softmax of the student's logits (approximate distribution)
        student_output_soft = F.softmax(student_logits, dim=2)
        # Calculate the reverse KL Divergence (KL(teacher_logits || student_logits))
        reverse_kl = F.kl_div(teacher_output_log_prob, student_output_soft, reduction="none").sum(-1)

        # forward
        student_output_log_prob = F.log_softmax(student_logits, dim=2)
        teacher_output_soft = F.softmax(teacher_logits, dim=2)
        # Calculate the reverse KL Divergence (KL(teacher_logits || student_logits))
        forward_kl = F.kl_div(student_output_log_prob, teacher_output_soft, reduction="none").sum(-1)

        kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
        kl_loss *= mask
        kl_loss = kl_loss.sum(-1).mean()
        return kl_loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # teacher forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                **inputs
            )
        teacher_logits = teacher_outputs.get("logits")
        del teacher_outputs

        # student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.get("logits")
        
        # distill loss
        kd_loss = self.cakld_loss(inputs['labels'], student_logits, teacher_logits, self.mean_prob)

        # task loss
        base_loss = self.label_smoother(student_outputs, inputs['labels'], shift_labels=True)

        kd_loss += base_loss

        if not return_outputs:
            del student_outputs

        del teacher_logits
        del student_logits

        return (kd_loss, student_outputs) if return_outputs else kd_loss