import json
from pathlib import Path


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, '_asdict'):
            return o._asdict()
        return json.JSONEncoder.default(self, o)


def save_arguments(arguments, path: Path):
    json.dump(vars(arguments), path.open('w'), indent=2, sort_keys=True, cls=JSONEncoder)
