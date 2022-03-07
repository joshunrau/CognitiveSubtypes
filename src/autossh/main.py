import argparse
import sys

from .core import CCServer, CICServer

def main():

    parser = argparse.ArgumentParser(prog="autossh")
    subparsers = parser.add_subparsers(title='servers')

    cc = subparsers.add_parser("cc", help="Compute Canada Server")
    cc.add_argument("--login", "-l", default=None, action='store_true')
    cc.add_argument("--download", "-d", nargs="*", help="file(s) to download from server, local download directory")
    cc.add_argument("--upload", "-u", nargs="*",help="file(s) to upload to server, directory on server where file(s) will be saved"),
    cc.set_defaults(cls=CCServer)

    cic = subparsers.add_parser("cic", help="CIC Server")
    cic.add_argument("--login", "-l", default=None, action='store_true')
    cic.add_argument("--download", "-d", nargs="*", help="file(s) to download from server, local download directory")
    cic.add_argument("--upload", "-u", nargs="*", help="file(s) to upload to server, directory on server where file(s) will be saved")
    cic.set_defaults(cls=CICServer)

    args = parser.parse_args(sys.argv[1:])

    n = 3 - [args.login, args.download, args.upload].count(None)
    if n != 1:
        raise ValueError(f"Expected one of 'l', 'd', 'u', but received {n} arguments")

    try:
        if len(args.download) not in range(1, 3):
            raise ValueError(f"Invalid number of arguments: {len(args.download)}")
    except TypeError:
        pass

    try:
        if len(args.upload) != 2:
            raise ValueError(f"Invalid number of arguments: {len(args.upload)}")
    except TypeError:
        pass

    server = args.cls(args)
    server.ssh()


if __name__ == main():
    main()
