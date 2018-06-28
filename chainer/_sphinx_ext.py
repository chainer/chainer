import sphinx


def setup(app):
    app.setup_extension('sphinx.ext.napoleon')

    def _parse_attributes_section(self, section):
        # type: (unicode) -> List[unicode]
        lines = []
        for _name, _type, _desc in self._consume_fields():
            if self._config.napoleon_use_ivar:
                if not _name.startswith('~') and self._obj:
                    _name = '~%s.%s' % (self._obj.__qualname__, _name)
                field = ':ivar %s: ' % _name  # type: unicode
                lines.extend(self._format_block(field, _desc))
                if _type:
                    lines.append(':vartype %s: %s' % (_name, _type))
            else:
                lines.extend(['.. attribute:: ' + _name, ''])
                fields = self._format_field('', '', _desc)
                lines.extend(self._indent(fields, 3))
                if _type:
                    lines.append('')
                    lines.extend(self._indent([':type: %s' % _type], 3))
                lines.append('')
        if self._config.napoleon_use_ivar:
            lines.append('')
        return lines

    sphinx.ext.napoleon.GoogleDocstring._parse_attributes_section = \
        _parse_attributes_section
