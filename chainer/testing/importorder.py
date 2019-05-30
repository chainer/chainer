import flake8_import_order


def _fullpath(modules, names):
    if not names:
        return modules[0]
    else:
        return '{}.{}'.format(modules[0], names[0])


class Hacking(flake8_import_order.styles.Style):
    """OpenStack Style Guideline"""

    def check(self):
        for error in super(Hacking, self).check():
            yield error
        for current in self.nodes:
            if isinstance(current, flake8_import_order.ClassifiedImport):
                for error in self._check_hacking(current):
                    yield error

    def _check_hacking(self, import_):
        # H304
        if import_.level > 0:
            message = (
                'No relative imports. \'{0}\' is a relative import'
            ).format(
                self._explain_import(import_)
            )
            yield flake8_import_order.styles.Error(
                import_.lineno, 'I304', message)

        # H303
        # wildcard import ('*' in import_.names)  is detected by flake8 (F403).

        # H301
        # len(import_.modules) >= 2 is detected by flake8 (E401).
        # We have to check importing multiple names
        if len(import_.names) >= 2:
            yield flake8_import_order.styles.Error(
                import_.lineno,
                'I301',
                'one import per line'
            )

    @staticmethod
    def import_key(import_):
        # H306
        return (import_.type, _fullpath(import_.modules, import_.names))
