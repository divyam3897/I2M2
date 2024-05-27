import torch.nn as nn
import torch.nn.functional as F
import torch



class InterModalityClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, exp_name):
        super().__init__()
        
        if "vqa" in exp_name: 
            self.model = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, output_dim),
                    ) 
        
        else:
            self.model = nn.Sequential(
                        nn.Linear(hidden_dim * 4, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, 2),
                    )

    def forward(self, x):
        return self.model(x)


class UnimodalClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, exp_name):
        super().__init__()
        if "vqa" in exp_name: 
            self.model = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, output_dim),
                    ) 
        
        else:
            if "image" in exp_name:
                self.model = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, 2),
                    )
            else:
                self.model = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, 2),
                    )

    def forward(self, x):
        return self.model(x)


class IntraModalityClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, exp_name):
        super().__init__()
        if "vqa" in exp_name: 
            self.image_model = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, output_dim),
                    ) 
        
        else:
            self.image_model = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, 2),
                    )
        if "vqa" in exp_name: 
            self.text_model = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, output_dim),
                    ) 
        
        else:
            self.text_model = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, 2),
                    )

    def forward(self, x_image, x_text):
        outputs_image = self.image_model(x_image)
        outputs_text = self.text_model(x_text)
        output_num = torch.log_softmax(outputs_image, dim=-1) +  \
                     torch.log_softmax(outputs_text, dim=-1)

        output_den = torch.logsumexp(output_num, dim=-1)
        outputs = output_num - output_den.unsqueeze(1)
        return outputs
    
class InterAndIntraModalityClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, exp_name):
        super().__init__()
        if "vqa" in exp_name: 
            self.cat_model = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, output_dim),
                    ) 
        
        else:
            self.cat_model = nn.Sequential(
                        nn.Linear(hidden_dim * 4, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, 2),
                    )
        if "vqa" in exp_name: 
            self.image_model = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, output_dim),
                    ) 
        
        else:
            self.image_model = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, 2),
                    )
        if "vqa" in exp_name: 
            self.text_model = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, output_dim),
                    ) 
        
        else:
            self.text_model = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.LayerNorm(hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, 2),
                    )
        self.inter = nn.Parameter(torch.tensor(1.)) 
        self.intra_1 = nn.Parameter(torch.tensor(1.))
        self.intra_2 = nn.Parameter(torch.tensor(1.))



    def forward(self, x, x_image, x_text):
        outputs_image = self.image_model(x_image)
        outputs_text = self.text_model(x_text)
        outputs_cat = self.cat_model(x)
        output_num = torch.log_softmax(outputs_image, dim=-1) +  \
                     torch.log_softmax(outputs_text, dim=-1) + \
                     torch.log_softmax(outputs_cat, dim=-1)
        output_den = torch.logsumexp(output_num, dim=-1)
        outputs = output_num - output_den.unsqueeze(1)
        return outputs, outputs_cat, outputs_image, outputs_text
