from audiomentations.core.transforms_interface import BaseWaveformTransform
from numpy.typing import NDArray
import random
import numpy as np
from torchvision.transforms.v2._transform import _RandomApplyTransform, Transform
from typing import Any, Callable, Dict, List, Tuple
from torch.nn.functional import one_hot
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._transform import _RandomApplyTransform, Transform
from torchvision.transforms.v2._utils import _parse_labels_getter, has_any, is_pure_tensor
import torch
class RandomMask(BaseWaveformTransform):
    

    supports_multichannel = True

    def __init__(
        self,
        p: float = 0.05,
        pr: float = 0.05,
        ):
        
        super().__init__(p)
        
        self.pr=pr
    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            num_samples = samples.shape[-1]
            self.parameters["t"] = random.randint(
                int(num_samples * self.min_band_part),
                int(num_samples * self.max_band_part),
            )
            self.parameters["t0"] = random.randint(
                0, num_samples - self.parameters["t"]
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        new_samples = samples.copy()
        
        mask = (np.random.rand(*samples.shape) < self.pr).astype(int)
        
        mask=np.cumsum(mask,axis=-1) % 2
        #print(mask.shape)      

        new_samples=new_samples*mask
        """if self.fade:
            fade_length = min(int(sample_rate * 0.01), int(t * 0.1))
            mask[0:fade_length] = np.linspace(1, 0, num=fade_length)
            mask[-fade_length:] = np.linspace(0, 1, num=fade_length)
        new_samples[..., t0 : t0 + t] *= mask"""
        
        return new_samples
    
class _BaseMixUpCutMix(Transform):
    def __init__(self, *, alpha: float = 1.0, num_classes: int, labels_getter="default") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

        self.num_classes = num_classes

        self._labels_getter = _parse_labels_getter(labels_getter)

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        needs_transform_list = self._needs_transform_list(flat_inputs)

        if has_any(flat_inputs, PIL.Image.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask):
            raise ValueError(f"{type(self).__name__}() does not support PIL images, bounding boxes and masks.")

        labels = self._labels_getter(inputs)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"The labels must be a tensor, but got {type(labels)} instead.")
        elif labels.ndim != 1:
            raise ValueError(
                f"labels tensor should be of shape (batch_size,) " f"but got shape {labels.shape} instead."
            )

        params = {
            "labels": labels,
            "batch_size": labels.shape[0],
            **self._get_params(
                [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
            ),
        }

        # By default, the labels will be False inside needs_transform_list, since they are a torch.Tensor coming
        # after an image or video. However, we need to handle them in _transform, so we make sure to set them to True
        needs_transform_list[next(idx for idx, inpt in enumerate(flat_inputs) if inpt is labels)] = True
        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

    def _check_image_or_video(self, inpt: torch.Tensor, *, batch_size: int):
        expected_num_dims = 5 if isinstance(inpt, tv_tensors.Video) else 4
        if inpt.ndim != expected_num_dims:
            raise ValueError(
                f"Expected a batched input with {expected_num_dims} dims, but got {inpt.ndim} dimensions instead."
            )
        if inpt.shape[0] != batch_size:
            raise ValueError(
                f"The batch size of the image or video does not match the batch size of the labels: "
                f"{inpt.shape[0]} != {batch_size}."
            )

    def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:
        label = one_hot(label, num_classes=self.num_classes)
        if not label.dtype.is_floating_point:
            label = label.float()
        return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))


#modified version of MixUp to handle unlabled 1D data
class MixUp(_BaseMixUpCutMix):
    """Apply MixUp to the provided batch of images and labels.

    Paper: `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``MixUp()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))  # type: ignore[arg-type]
        
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        lam = params["lam"]
        lam=0.7
        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=lam)
        else:
            print(inpt.shape)
            
            return inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))
        