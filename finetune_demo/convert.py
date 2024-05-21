import json
from typing import Union
from pathlib import Path
import sys

data_src = "data/AdvertiseGen"

if sys.argv[1]:
    data_src = sys.argv[1]

data_fix = data_src + "_fix"


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _mkdir(dir_name: Union[str, Path]):
    dir_name = _resolve_path(dir_name)
    if not dir_name.is_dir():
        dir_name.mkdir(parents=True, exist_ok=False)


# 数据集格式化
def convert_adgen(data_dir: Union[str, Path], save_dir: Union[str, Path]):

    def _convert(in_file: Path, out_file: Path):
        _mkdir(out_file.parent)
        with open(in_file, encoding='utf-8') as fin:
            with open(out_file, 'wt', encoding='utf-8') as fout:
                for line in fin:
                    dct = json.loads(line)
                    sample = {
                        'conversations': [{
                            'role': 'user',
                            'content': dct['content']
                        }, {
                            'role': 'assistant',
                            'content': dct['summary']
                        }]
                    }
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

    data_dir = _resolve_path(data_dir)
    save_dir = _resolve_path(save_dir)

    train_file = data_dir / 'train.json'
    if train_file.is_file():
        out_file = save_dir / train_file.relative_to(data_dir)
        _convert(train_file, out_file)

    dev_file = data_dir / 'dev.json'
    if dev_file.is_file():
        out_file = save_dir / dev_file.relative_to(data_dir)
        _convert(dev_file, out_file)


convert_adgen(data_src, data_fix)
