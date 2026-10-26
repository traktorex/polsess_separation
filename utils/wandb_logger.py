"""Weights & Biases logging wrapper for experiment tracking."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger:
    """Wrapper for Weights & Biases experiment tracking."""

    def __init__(
        self,
        project: str = "polsess-separation",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
        enabled: bool = True,
        logger: Optional[logging.Logger] = None,
        run: Optional[Any] = None,
        upload_checkpoints: bool = True,
        provenance: Optional[Dict[str, Any]] = None,
        resume_id: Optional[str] = None,
    ):
        """Initialize W&B logger. If run is provided, uses it instead of creating new one.

        Args:
            upload_checkpoints: If False, skip uploading model artifacts to W&B.
                Set to False during sweeps to avoid storage bloat.
            provenance: Optional run manifest (see ``collect_run_manifest``);
                attached to the run config under a ``"provenance"`` key so the
                git SHA / env / GPU that produced the run are visible in W&B.
            resume_id: Optional W&B run id to reconnect to (survey gap 14). When
                set (and no existing ``run`` is passed) the run is created with
                ``id=<resume_id>, resume="must"`` so a resumed training continues
                the *same* W&B run instead of orphaning a fresh one.
        """
        self.logger = logger or logging.getLogger("polsess")
        self.enabled = enabled and WANDB_AVAILABLE
        self.upload_checkpoints = upload_checkpoints
        self.run = None

        if not self.enabled:
            if not WANDB_AVAILABLE:
                self.logger.warning(
                    "W&B not available"
                )
            else:
                self.logger.info("W&B logging disabled")
            return

        try:
            # Use existing run if provided, otherwise create new one
            if run is not None:
                self.run = run
                self.logger.info("Using existing W&B run")
                # Existing-run path (sweeps): attach provenance to the run config
                # after the fact. allow_val_change since the config already exists.
                if provenance:
                    try:
                        self.run.config.update(
                            {"provenance": provenance}, allow_val_change=True
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to attach provenance to W&B run config: {e}"
                        )
            else:
                config_dict = {}
                if config:
                    if hasattr(config, "data"):
                        config_dict = {
                            "data": asdict(config.data),
                            "model": asdict(config.model),
                            "training": asdict(config.training),
                        }
                    else:
                        config_dict = config

                if provenance:
                    # Nest under its own key so it can't collide with a config field.
                    config_dict = {**config_dict, "provenance": provenance}

                init_kwargs = dict(
                    project=project,
                    entity=entity,
                    name=run_name,
                    config=config_dict,
                    # Console logs: by default the SDK writes one ``output.log``
                    # per process, so a resumed run *replaces* the earlier
                    # session's console log in the W&B "Logs" tab. Multipart
                    # mode writes timestamped parts under ``logs/`` instead, so
                    # resumed sessions append. With both chunk limits at 0 the
                    # parts would only upload at run finish, hence the time
                    # rollover: each part uploads when closed, keeping the tab
                    # near-live (a tqdm bar spanning a boundary just freezes
                    # its last line in the earlier part).
                    settings=wandb.Settings(
                        console_multipart=True,
                        console_chunk_max_seconds=600,
                    ),
                )
                if resume_id:
                    init_kwargs["id"] = resume_id
                    init_kwargs["resume"] = "must"
                    self.logger.info(f"Resuming W&B run id={resume_id} (resume='must')")
                else:
                    init_kwargs["resume"] = "allow"
                self.run = wandb.init(**init_kwargs)

        except Exception as e:
            self.logger.error(f"Failed to initialize W&B: {e}")
            self.logger.warning("Continuing without W&B logging")
            self.enabled = False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B."""
        if not self.enabled or not self.run:
            return

        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to W&B: {e}")

    def log_model(self, model_path: str, name: Optional[str] = None):
        """Save model checkpoint as W&B artifact."""
        if not self.enabled or not self.run or not self.upload_checkpoints:
            return

        try:
            model_path = Path(model_path)
            if not model_path.exists():
                self.logger.warning(f"Model file not found: {model_path}")
                return

            artifact_name = name or model_path.stem
            # Include run name to avoid all runs versioning the same artifact
            if self.run.name and not name:
                artifact_name = f"{artifact_name}-{self.run.name}"
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(str(model_path))
            self.run.log_artifact(artifact)

            self.logger.info(f"Logged model artifact: {artifact_name}")

        except Exception as e:
            self.logger.warning(f"Failed to log model to W&B: {e}")

    def finish(self):
        """Finish W&B run gracefully."""
        if not self.enabled or not self.run:
            return

        try:
            wandb.finish()
            self.logger.info("W&B run finished")
        except Exception as e:
            self.logger.warning(f"Error finishing W&B run: {e}")
