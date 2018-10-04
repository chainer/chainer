import sphinx


def qualify_name(attr_name, klass):
    if klass and '.' not in attr_name:
        if attr_name.startswith('~'):
            attr_name = attr_name[1:]
        try:
            q = klass.__qualname__
        except AttributeError:
            q = klass.__name__
        return '~%s.%s' % (q, attr_name)
    return attr_name


def setup(app):
    app.setup_extension('sphinx.ext.napoleon')

    def _parse_attributes_section(self, section):
        # type: (unicode) -> List[unicode]
        lines = []
        for _name, _type, _desc in self._consume_fields():
            if self._config.napoleon_use_ivar:
                _name = qualify_name(_name, self._obj)  # Added this line
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
