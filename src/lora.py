import torch
from functools import partial

class LoRAConfig:
    def __init__(
        self,
        model,
        r = 8,
        alpha = 16,
        dropout = 0.05,
        query = True,
        key = False,
        value = True,
        projection = False,
        mlp = False,
        head = False
        ) -> None:
        self.model = model
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.query = query
        self.key = key
        self.value = value
        self.projection = projection
        self.mlp = mlp
        self.head = head
        
        # Add support for other models by implementing the iterate_through_{model}
        # function and add that method into this hashmap
        self.modify_layers_executor = {
            "BertModelForSequenceClassification":self.__iterate_through_bert,
            "DistilBertForSequenceClassification":self.__iterate_through_distilbert
        }
        
        # Apply  LoRA layers
        self.__apply_lora_to_model()
        
    def __apply_lora_to_model(self, ):
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Assigning LoRA layers on every linear layer in the transformer model
        self.assign_lora = partial(LinearWithLora, rank=self.r, alpha=self.alpha)
        
        # Call the right function to modify layers
        self.modify_layers_executor[self.model.__class__.__name__]()
        
        # Emit stats
        total_params = sum(p.numel() for p in self.model.parameters())
        lora_params = self.count_lora_parameters()
        print(f"Trainable parameters: {lora_params} | Percentage: {round(lora_params*100/total_params, 3)}%")
        
    def count_lora_parameters(self, ):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def __iterate_through_bert(self, ):
        for layer in self.model.children():
            if self.query:
                layer.attention.q_lin = self.assign_lora(layer.attention.query)
            if self.key:
                layer.attention.k_lin = self.assign_lora(layer.attention.key)
            if self.value:
                layer.attention.v_lin = self.assign_lora(layer.attention.value)
            if self.projection:
                layer.attention.out_lin = self.assign_lora(layer.attention.output)
            if self.mlp:
                layer.ffn.lin1 = self.assign_lora(layer.ffn.lin1)
                layer.ffn.lin2 = self.assign_lora(layer.ffn.lin2)
        if self.head:
            self.model.pre_classifier =self.assign_lora(self.model.pre_classifier)
            self.model.classifier = self.assign_lora(self.model.classifier)
            
    def __iterate_through_distilbert(self, ):
        for layer in self.model.distilbert.transformer.layer:
            if self.query:
                layer.attention.q_lin = self.assign_lora(layer.attention.q_lin)
            if self.key:
                layer.attention.k_lin = self.assign_lora(layer.attention.k_lin)
            if self.value:
                layer.attention.v_lin = self.assign_lora(layer.attention.v_lin)
            if self.projection:
                layer.attention.out_lin = self.assign_lora(layer.attention.out_lin)
            if self.mlp:
                layer.ffn.lin1 = self.assign_lora(layer.ffn.lin1)
                layer.ffn.lin2 = self.assign_lora(layer.ffn.lin2)
        if self.head:
            self.model.pre_classifier = self.assign_lora(self.model.pre_classifier)
            self.model.classifier = self.assign_lora(self.model.classifier)
    
        
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        
        # The A matrix is scaled by the stdev which is obtainde by sqrt(rank) so that the 
        # values are not very large
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        
        #  Initially set to 0, since before beginning of training LoRA layer should
        # not affect the model weights. If B=0, then A.B = 0
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    
    
class LinearWithLora(torch.nn.Module):
    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            in_dim=linear.in_features, 
            out_dim=linear.out_features, 
            rank=rank, 
            alpha=alpha,
            )
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)
    