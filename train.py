import argparse
from pathlib import Path

from yukarin.config import create_from_json
from yukarin.trainer import create_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json_path', type=Path)
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()

    config = create_from_json(arguments.config_json_path)
    arguments.output.mkdir(exist_ok=True)

    trainer = create_trainer(config, arguments.output)
    trainer.run()


if __name__ == '__main__':
    main()
