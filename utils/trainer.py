import torch
import datasets
from typing import Optional

from torch import nn
from transformers import Trainer

from torch.utils.data import RandomSampler, SequentialSampler

from sampler.ClassAwareSampler import ClassAwareSampler
from sampler.ClassPrioritySampler import ClassPrioritySampler
from sampler.MixedPrioritizedSampler import MixedPrioritizedSampler


class CustomTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        task = "ClassPriority"
        if task == "ClassAware":
            return ClassAwareSampler(
                data_source=self.train_dataset,
                seed=seed
            )
        elif task == "ClassPriority":
            return ClassPrioritySampler(
                dataset=self.train_dataset,
            )
        elif task == "MixedPrioritized":
            return MixedPrioritizedSampler(
                dataset=self.train_dataset,
            )
        elif task == "Sequential":
            return SequentialSampler(self.train_dataset)
        elif task == "Random":
            return RandomSampler(self.train_dataset)
        else:
            print("Error! Wrong sampler name!")
            return None

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.get("labels")
    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)
    #     loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
    #     loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
