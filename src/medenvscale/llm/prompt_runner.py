from __future__ import annotations

from pathlib import Path
from typing import Any

class PromptRunner:
    def __init__(self, prompt_dir: str | Path) -> None:
        self.prompt_dir = Path(prompt_dir)
        try:
            from jinja2 import Environment, FileSystemLoader, StrictUndefined

            self.env = Environment(
                loader=FileSystemLoader(str(prompt_dir)),
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
            )
        except ModuleNotFoundError:
            self.env = None

    def render(self, template_name: str, **kwargs: Any) -> str:
        if self.env is not None:
            template = self.env.get_template(template_name)
            return template.render(**kwargs)
        template = (self.prompt_dir / template_name).read_text(encoding="utf-8")
        rendered = template
        for key, value in kwargs.items():
            rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
        return rendered
