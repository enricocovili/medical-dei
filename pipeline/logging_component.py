from __future__ import annotations

import logging


class PipelineEventLogger:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def pipeline_started(self, run_mode: str) -> None:
        self._logger.info("Pipeline started — mode: %s", run_mode)

    def stage_started(self, stage: str) -> None:
        self._logger.info("[%s] Stage started", stage.upper())

    def stage_completed(self, stage: str, count: int) -> None:
        self._logger.info("[%s] Stage completed — %d image(s) processed", stage.upper(), count)

    def loading_model(self, model_name: str) -> None:
        self._logger.info("[INIT] Loading model: %s", model_name)

    def model_loaded(self, model_name: str) -> None:
        self._logger.info("[INIT] Model ready: %s", model_name)

    def processing_image(self, image_name: str, stage: str | None = None) -> None:
        if stage:
            self._logger.info("[%s] Processing: %s", stage.upper(), image_name)
        else:
            self._logger.info("Processing image: %s", image_name)

    def alarm_triggered(self, image_name: str, motivation: str) -> None:
        self._logger.warning("[POSTPROCESS] Alarm on %s: %s", image_name, motivation)

    def fallback_prompt_attempted(self, image_name: str, prompt: str) -> None:
        self._logger.info(
            "[SAM3] No boxes found for %s — retrying with fallback prompt: '%s'",
            image_name,
            prompt,
        )

    def image_completed(
        self,
        image_name: str,
        sam_time: float,
        morph_time: float,
        deid_time: float,
    ) -> None:
        self._logger.info(
            "Completed %s | sam3=%.3fs erosion_diffusion=%.3fs deid=%.3fs",
            image_name,
            sam_time,
            morph_time,
            deid_time,
        )

    def report_written(self, output_file: str, records: int) -> None:
        self._logger.info("[REPORT] Wrote %d record(s) to %s", records, output_file)
