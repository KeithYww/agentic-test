# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path

from common import utils
from common.prompt_base import PromptBase

logger = utils.get_logger("test_case_classification_prompt")
PROMPTS_ROOT = "system_prompts"


def _get_prompts_root() -> Path:
    return Path(__file__).resolve().parent.joinpath(PROMPTS_ROOT)


class TestCaseClassificationSystemPrompt(PromptBase):
    """Thin instructions for the main orchestrating agent: just call the classification tool."""

    def get_script_dir(self) -> Path:
        return _get_prompts_root()

    def __init__(self, template_file_name: str = "main_prompt_template.txt"):
        super().__init__(template_file_name)

    def get_prompt(self) -> str:
        logger.info("Generating test case classification main system prompt")
        return self.template


class TestCaseClassificationBatchPrompt(PromptBase):
    """Prompt for the sub-agent that classifies one batch of test cases."""

    def get_script_dir(self) -> Path:
        return _get_prompts_root()

    def __init__(self, template_file_name: str = "batch_prompt_template.txt"):
        super().__init__(template_file_name)

    def get_prompt(self) -> str:
        logger.info("Generating test case classification batch system prompt")
        return self.template
