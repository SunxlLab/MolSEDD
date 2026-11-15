"""Conditional (prefix / suffix constrained) generation with a trained SEDD model.

Conditioning is done by *projection*: at every denoising step the fixed prefix and
suffix tokens are written back into the sampled sequence (``proj_fun``, supported
natively by :func:`sedd.sampling.get_pc_sampler`).  No extra parameters and no
fine-tuning are required.
"""

from typing import List, Optional

import torch

from sedd.load_model import load_model
from sedd.sampling import get_pc_sampler

from .vocab import DatasetSpec, get_spec


class MolecularGenerator:
    def __init__(
        self,
        model_path: str,
        dataset: str = "moses",
        seq_len: Optional[int] = None,
        device: Optional[str] = None,
        spec: Optional[DatasetSpec] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # ``spec`` lets the per-dataset scripts pass their own frozen vocabulary
        # instead of looking it up by name.
        self.spec: DatasetSpec = spec or get_spec(dataset)
        self.seq_len = seq_len or self.spec.seq_len
        self.model, self.graph, self.noise = load_model(model_path, self.device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    def generate(
        self,
        prefix: str = "",
        suffix: str = "$",
        batch_size: int = 200,
        steps: int = 1024,
    ) -> List[str]:
        """Sample ``batch_size`` molecules that all start with ``prefix``.

        ``suffix`` is pinned to the tail of the sequence (``"$"`` = end token).
        Returns the decoded SMILES strings.
        """
        prefix_ids = self.spec.encode(self.spec.tokenize(prefix))
        suffix_ids = self.spec.encode(self.spec.tokenize(suffix))
        input_ids = prefix_ids + suffix_ids

        if len(input_ids) > self.seq_len:
            # The prefix no longer fits in the model context: nothing to do.
            return []

        input_locs = list(range(len(prefix_ids))) + list(
            range(self.seq_len - len(suffix_ids), self.seq_len)
        )
        fixed = torch.tensor(input_ids, device=self.device)[None].repeat(batch_size, 1)

        def proj_fun(x):
            x[:, input_locs] = fixed
            return x

        sampling_fn = get_pc_sampler(
            self.graph,
            self.noise,
            (batch_size, self.seq_len),
            "analytic",
            steps,
            device=self.device,
            proj_fun=proj_fun,
        )
        samples = proj_fun(sampling_fn(self.model))
        return [self.spec.decode(row.tolist()) for row in samples.cpu().numpy()]
