#!/usr/bin/env python

import argparse
import os

if __name__ == '__main__':
    # configure arg parser
    parser = argparse.ArgumentParser(description='Makes local non-versioned config.py for working with scripts ouside of own-cloud. Generates valid config.py with the correct local paths set', prog='make_config')
    positional_args = parser.add_argument_group('Positional arguments')
    optional_args = parser.add_argument_group('Optional arguments')
    positional_args.add_argument('Root', nargs='?', help='The absolute path to the local own-cloud root script directory (processing)')
    optional_args.add_argument('-f', '--force', action='store_true', default=False, help='Force overwrite of existing config')
    # run the arg parser
    arguments = parser.parse_args()
    existing_config = False
    try:
        import config
        existing_config = True
    except ImportError:
        pass
    if not arguments.Root:
        raise ValueError('Must provide Root directory. See help')
    if not os.path.exists(arguments.Root):
        raise ValueError('Path {} does not exist on local machine'.format(arguments.Root))
    if existing_config and not arguments.force:
        raise RuntimeError('config.py already exists. cannot overwrite without force option. see help.')
    # make the config.py
    with open('config.py', 'w') as fh:
        fh.write('script_root="{}"\n'.format(arguments.Root))
    print('config.py written')
