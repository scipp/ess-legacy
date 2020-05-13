#!/usr/bin/env python

import argparse
import os

if __name__ == '__main__':
    # configure arg parser
    parser = argparse.ArgumentParser(
        description=
        'Makes local non-versioned dataconfig.py for working with LOKI data. test data directory found git@github.com:ess-dmsc-dram/loki_tube_scripts.git assumed to be checked out locally. Generates valid dataconfig.py with the correct local paths set',
        prog='make_config')
    positional_args = parser.add_argument_group('Positional arguments')
    optional_args = parser.add_argument_group('Optional arguments')
    positional_args.add_argument(
        'Root',
        nargs='?',
        help='The absolute path to the local Loki test data directory')
    optional_args.add_argument('-f',
                               '--force',
                               action='store_true',
                               default=False,
                               help='Force overwrite of existing config')
    # run the arg parser
    arguments = parser.parse_args()
    existing_config = False
    try:
        import dataconfig
        existing_config = True
    except ImportError:
        pass
    if not arguments.Root:
        raise ValueError('Must provide Root directory. See help')
    if not os.path.exists(arguments.Root):
        raise ValueError('Path {} does not exist on local machine'.format(
            arguments.Root))
    if existing_config and not arguments.force:
        raise RuntimeError(
            'config.py already exists. cannot overwrite without force option. see help.'
        )
    # make the config.py
    with open('dataconfig.py', 'w') as fh:
        fh.write('data_root="{}"\n'.format(arguments.Root))
    print('dataconfig.py written')
