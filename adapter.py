import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from typing import List, Set


class InstructionFingerprint(torch.nn.Module):
    def __init__(self, emb: torch.nn.Module, all_trainable_input_ids: List[int], inner_dim=128):
        super().__init__()
        self.orig_emb = emb
        self.all_trainable_input_ids = all_trainable_input_ids
        self.trainable_emb = torch.nn.Embedding(len(all_trainable_input_ids), self.orig_emb.weight.size(1))
        with torch.no_grad():
            self.trainable_emb.weight.copy_(emb.weight[all_trainable_input_ids])
        self.A = torch.nn.Linear(self.orig_emb.weight.size(1), inner_dim)
        self.B = torch.nn.Linear(inner_dim, self.orig_emb.weight.size(1))
        with torch.no_grad():
            # 0 init
            self.A.weight.zero_(); self.A.bias.zero_()
            self.B.weight.zero_(); self.B.bias.zero_()
        self.cast_dtype()
    
    @property
    def weight(self):
        return self.orig_emb.weight

    @torch.no_grad()
    def cast_dtype(self):
        dtype = self.orig_emb.weight.dtype
        for param in self.parameters():
            param.data = param.data.to(dtype=dtype)

    def forward(self, input): # from nn.Embedding
        """
        input (N, L)
        for each input, we need to find out whether it is in all_trainable_input_ids
            if it is, we use @trainable_emb
            if not, we use @orig_emb
        """
        N, L = input.size()
        all_trainable_input_ids_tensor = torch.tensor(self.all_trainable_input_ids).to(device=input.device, dtype=input.dtype)
        # (N, L) Compute a mask of the same shape as input where an element is True if it's in all_trainable_input_ids
        mask = (input.unsqueeze(-1) == all_trainable_input_ids_tensor).any(-1)

        # For values in input that are in all_trainable_input_ids
        # (n, ) where n = mask.sum(), each ele is the indices wrt all_trainable_input_ids
        indices = (input[mask].unsqueeze(-1) == all_trainable_input_ids_tensor).max(-1).indices

        # For values in input that are in all_trainable_input_ids, compute embeddings using trainable_emb and do adapter
        embeddings_from_trainable = self.B(self.A(self.trainable_emb(indices))) + self.orig_emb(input[mask])
        # For values not in all_trainable_input_ids
        embeddings_from_orig = self.orig_emb(input[~mask])
        
        # Create an empty tensor of the required shape to hold results
        output = torch.empty(N, L, self.orig_emb.weight.size(1), device=input.device, dtype=self.orig_emb.weight.dtype)

        # Use the mask to place the computed embeddings in the right positions
        output[mask] = embeddings_from_trainable
        output[~mask] = embeddings_from_orig
        return output
    
    @torch.no_grad()
    def merge(self):
        """
        merge trainable_emb into orig_emb
        """
        self.orig_emb.weight[self.all_trainable_input_ids] = self.trainable_emb.weight


def inject_adapter_to(model: AutoModelForCausalLM, all_trainable_input_ids: Set[int], trained_adapter=None, inner_dim=16):
    def find_emb_and_replace(model, trained_adapter):
        """
        find embedding layer and replace it with InstructionFingerprint/trained_adapter
        return the replaced adapter
        """
        emb_attr_str = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding) and module is model.get_input_embeddings():
                emb_attr_str = name
        assert emb_attr_str is not None, "Cannot find embedding layer"
        # e.g. emb_attr_str = "transformer.wte"
        emb_attr_lists = emb_attr_str.split(".")
        model_so_far = model
        for attr in emb_attr_lists[:-1]:
            model_so_far = getattr(model_so_far, attr)
        if trained_adapter is not None:
            replaced_adpter = trained_adapter
            setattr(model_so_far, emb_attr_lists[-1], trained_adapter)
        else:
            replaced_adpter = InstructionFingerprint(model.get_input_embeddings(), list(all_trainable_input_ids), inner_dim=inner_dim)
            setattr(model_so_far, emb_attr_lists[-1], replaced_adpter)
        assert isinstance(model.get_input_embeddings(), InstructionFingerprint)
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True
        model.get_input_embeddings().orig_emb.weight.requires_grad = False
        num_of_trainable_params = sum(
            [p.numel() for p in model.parameters() if p.requires_grad]
        )
        print("Replacing", emb_attr_lists, 
            f"({model.get_input_embeddings().orig_emb.weight.shape}) with InstructionFingerprint ({num_of_trainable_params})")
        return replaced_adpter
    
    for param in model.parameters():
        param.requires_grad = False
    
    if hasattr(model, "get_encoder"): # seq 2 seq
        assert model.get_encoder().get_input_embeddings() is model.get_decoder().get_input_embeddings(), "Only support shared embedding for now"
        replaced_adapter = find_emb_and_replace(model.get_encoder(), trained_adapter=trained_adapter)
        find_emb_and_replace(model.get_decoder(), trained_adapter=replaced_adapter)
        assert model.get_encoder().get_input_embeddings() is model.get_decoder().get_input_embeddings()
    else:
        find_emb_and_replace(model, trained_adapter=trained_adapter)
    return model
        
def unwrap_adapter(model):
    def find_emb_and_restore(model):
        instruction_emb = model.get_input_embeddings()
        instruction_emb.merge() # merge trainable_emb to orig_emb
        emb_attr_str = None
        for name, module in model.named_modules():
            if isinstance(module, InstructionFingerprint) and module is model.get_input_embeddings():
                emb_attr_str = name
        model_so_far = model
        emb_attr_lists = emb_attr_str.split(".")
        for attr in emb_attr_lists[:-1]:
            model_so_far = getattr(model_so_far, attr)
        setattr(model_so_far, emb_attr_lists[-1],
                instruction_emb.orig_emb)
        assert isinstance(model.get_input_embeddings(), torch.nn.Embedding)
        return instruction_emb

    if hasattr(model, "get_encoder"): # seq 2 seq
        assert model.get_encoder().get_input_embeddings() is model.get_decoder().get_input_embeddings()
        instruction_emb = find_emb_and_restore(model.get_encoder())
        assert instruction_emb is find_emb_and_restore(model.get_decoder())
        assert isinstance(model.get_encoder().get_input_embeddings(), torch.nn.Embedding) and isinstance(model.get_decoder().get_input_embeddings(), torch.nn.Embedding)
    else:
        instruction_emb = find_emb_and_restore(model)
    return model, instruction_emb