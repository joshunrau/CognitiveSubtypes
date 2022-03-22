import argparse
import subprocess as sp

from abc import ABC, abstractmethod
from shutil import which

from .config import Config

class Server(ABC, Config):

    def __init__(self, args: argparse.Namespace):

        if args.login:
            self.ssh = self.login
        elif args.download is not None:
            self.file = args.download[0]
            if len(args.download) == 2:
                self.dl_dir = args.download[1]
            self.ssh = self.download
        elif args.upload is not None:
            self.file = args.upload[0]
            self.dl_dir = args.upload[1]
            self.ssh = self.upload

        if which("sshpass") is None:
            raise Exception("Cannot find sshpass in path, make sure it is installed on system.")
        
        if self.dl_dir is None:
            raise ValueError("Download directory unspecified in 'autossh/config.py'")
        elif self.usr is None:
            raise ValueError("Username unspecified in 'autossh/config.py'")
        elif self.pwd is None:
            raise ValueError("pwd unspecified in 'autossh/config.py'")

    @property
    @abstractmethod
    def url(self):
        pass

    @property
    @abstractmethod
    def usr(self):
        pass

    @property
    @abstractmethod
    def pwd(self):
        pass

    @abstractmethod
    def login(self):
        pass

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def upload(self):
        pass

    @staticmethod
    def check_sshpass():
        if which("sshpass") is None:
            raise Exception("Cannot find sshpass in path, make sure it is installed on system.")


class CCServer(Server):

    url = "narval.computecanada.ca"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    @property
    def usr(self):
        return self.usernames["CC"]
    
    @property
    def pwd(self):
        return self.passwords["CC"]

    def login(self):
        sp.run(f"sshpass -p {self.pwd} ssh {self.usr}@{self.url}", shell=True)

    def download(self):
        sp.run(f"sshpass -p {self.pwd} rsync {self.usr}@{self.url}:{self.file} {self.dl_dir}", shell=True)

    def upload(self):
        sp.run(f"sshpass -p {self.pwd} rsync {self.file} {self.usr}@{self.url}:{self.dl_dir}", shell=True)


class CICServer(Server):

    url = "ps395560.dreamhostps.com"

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
    
    @property
    def usr(self):
        return self.usernames["CIC"]
    
    @property
    def pwd(self):
        return self.passwords["CIC"]

    def login(self):
        sp.run(f"sshpass -p {self.pwd} ssh -p 8764 {self.usr}@{self.url}", shell=True)

    def download(self):
        sp.run(f"sshpass -p {self.pwd} rsync -avz -e 'ssh -p 8764' {self.usr}@{self.url}:{self.file} {self.dl_dir}", shell=True)

    def upload(self):
        sp.run(f"sshpass -p {self.pwd} rsync -avz -e 'ssh -p 8764' {self.file} {self.usr}@{self.url}:{self.dl_dir}", shell=True)
