#!/usr/bin/env python

from six.moves.urllib import request


def main():
    request.urlretrieve(
        'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz',
        'tasks_1-20_v1-2.tar.gz')


if __name__ == '__main__':
    main()
