import logging
import os

# Keep library noise out of the command-line output. This has to run before the
# Hugging Face libraries and PyTorch are imported, which is why it lives here.
# Set the variables in your shell to override them.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
# PyTorch prints a macOS-only note on import that doesn't affect this project.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").disabled = True
