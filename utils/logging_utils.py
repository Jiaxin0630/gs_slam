import rich

_log_styles = {
    "GS-SLAM": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "    tracking results": "bold yellow",
    "    mapping results": "bold purple4",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="GS-SLAM",flush=False):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args, flush = flush)
