# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path

from common import utils
from common.prompt_base import PromptBase

logger = utils.get_logger("reviewer.agent")
PROMPTS_ROOT = "system_prompts"


def _get_prompts_root() -> Path:
    return Path(__file__).resolve().parent.joinpath(PROMPTS_ROOT)


class RequirementsReviewSystemPrompt(PromptBase):
    """
    Loads a prompt template for main orchestrator instructions.
    """

    def get_script_dir(self) -> Path:
        return _get_prompts_root()

    def __init__(self, template_file_name: str = "main_prompt_template.txt"):
        super().__init__(template_file_name)

    def get_prompt(self) -> str:
        """Returns the prompt as a string."""
        logger.info("Generating main requirements reviewer system prompt")
        return self.template


class RequirementsReviewContentPrompt(PromptBase):
    """
    Prompt for the sub-agent that reviews requirement document content.
    """

    def get_script_dir(self) -> Path:
        return _get_prompts_root()

    def __init__(self, template_file_name: str = "review_with_attachments_prompt.txt"):
        super().__init__(template_file_name)

    def get_prompt(self) -> str:
        """Returns the prompt as a string."""
        logger.info("Generating system prompt for sub-agent which performs requirements review")
        return self.template
