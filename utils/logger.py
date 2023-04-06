from collections import OrderedDict
from loguru import logger


class DialogLogger():
    def __init__(self, order, column_width=64, h_space=8):
        self.order = order
        self.column_width = column_width
        self.h_space = h_space

    def _multi_column_log(self, print_func, **columns):
        assert all([name in self.order for name in columns])

        name_lens = OrderedDict(
            [(name, len(name) + 2) for name in self.order if name in columns]
        )
        first_line = True

        columns = {k: str(v) for k, v in columns.items()}

        while columns:
            line = ""

            for name, length in name_lens.items():
                if name not in columns:
                    line += " " * (self.column_width + self.h_space)
                    continue

                if columns[name].startswith("\n"):
                    columns[name] = columns[name][1:]
                    line += " " * (self.column_width + self.h_space)
                    continue

                header = f"{name}: " if first_line else (" " * length)

                columns[name] = header + columns[name]
                crop = columns[name][:self.column_width].split("\n")[0]
                columns[name] = columns[name][len(crop):]
                columns[name] = columns[name][columns[name].startswith("\n"):]

                line += crop.ljust(self.column_width + self.h_space)

                if not columns[name]:
                    columns.pop(name)

            first_line = False
            print_func(line)
        print_func("------")

    def info(self, **columns):
        self._multi_column_log(logger.info, **columns)
