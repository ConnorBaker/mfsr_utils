from pathlib import Path

from torchvision.datasets.utils import (  # type: ignore[import]
    download_and_extract_archive,
    extract_archive,
)
from typing_extensions import ClassVar, Protocol


class Downloadable(Protocol):
    url: ClassVar[str]
    filename: ClassVar[str]
    dirname: ClassVar[str]
    mirrors: ClassVar[list[str]]

    @classmethod
    def download_to(cls, data_dir: Path) -> None:
        dir = data_dir / cls.dirname
        file = data_dir / cls.filename
        urls = [cls.url] + cls.mirrors

        if dir.exists():
            return

        if file.exists():
            extract_archive(from_path=file.as_posix(), to_path=dir.as_posix())
            return

        while not len(urls) == 0:
            url = urls.pop()
            try:
                download_and_extract_archive(
                    url,
                    download_root=data_dir.as_posix(),
                    extract_root=dir.as_posix(),
                    filename=cls.filename,
                )
                return
            except Exception as e:
                raise Exception(f"Could not download and extract dataset: {e}")

        if not file.exists():
            raise Exception("Could not download dataset")
