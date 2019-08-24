import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path
from pprint import pprint

import tqdm

from yukarin import AcousticConverter
from yukarin.config import create_from_json as create_config
from yukarin.param import AcousticParam
from yukarin.utility.json_utility import save_arguments

base_acoustic_param = AcousticParam()

parser = argparse.ArgumentParser()
parser.add_argument('--input_glob', '-i')
parser.add_argument('--output', '-o', type=Path)
parser.add_argument('--vc_model', '-vcm', type=Path)
parser.add_argument('--vc_config', '-vcc', type=Path)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--ignore_feature', nargs='+', default=['sp', 'ap'])
parser.add_argument('--enable_overwrite', action='store_true')
arguments = parser.parse_args()


def convert_feature(path: Path, acoustic_converter: AcousticConverter):
    out = Path(arguments.output, path.stem + '.npy')
    if out.exists() and not arguments.enable_overwrite:
        return

    in_feature = acoustic_converter.load_acoustic_feature(path)
    out_feature = acoustic_converter.convert(in_feature)
    out_feature = acoustic_converter.decode_spectrogram(out_feature)

    # save
    out_feature.save(path=out, ignores=arguments.ignore_feature)


def main():
    pprint(vars(arguments))

    arguments.output.mkdir(exist_ok=True)
    save_arguments(arguments, arguments.output / 'arguments.json')

    config = create_config(arguments.vc_config)
    acoustic_converter = AcousticConverter(config, arguments.vc_model, gpu=arguments.gpu)

    paths = [Path(p) for p in glob.glob(arguments.input_glob)]

    pool = multiprocessing.Pool()
    it = pool.imap(partial(convert_feature, acoustic_converter=acoustic_converter), paths)
    list(tqdm.tqdm(it, total=len(paths)))


if __name__ == '__main__':
    main()
