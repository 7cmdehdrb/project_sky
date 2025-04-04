from setuptools import find_packages, setup
import os
import sys
from glob import glob

package_name = "fcn_network"

resource_path = os.path.join(os.path.dirname(__file__), "resource")
file_names = os.listdir(resource_path)

valid_extensions = (".json", ".yaml", ".txt", ".pt", ".pth")
filtered_files = [f for f in file_names if f.endswith(valid_extensions)]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name]
            + [f"resource/{file}" for file in filtered_files],
        ),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="min",
    maintainer_email="7cmdehdrb@naver.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    # tests_require=['pytest'],
    entry_points={
        "console_scripts": [
            "fcn_server = fcn_network.fcn_server:main",
            "pointcloud_grid_identifier_server = fcn_network.pointcloud_grid_identifier_server:main",
        ],
    },
)
