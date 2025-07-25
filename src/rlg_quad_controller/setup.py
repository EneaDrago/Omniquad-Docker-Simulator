from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rlg_quad_controller'

data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # (os.path.join('share', package_name, 'models'), glob('models/*/*.pth')),
        # (os.path.join('share', package_name, 'models'), glob('models/*/*.yaml'))
    ]

def package_files(data_files, directory_list):
    paths_dict = {}
    for directory in directory_list:
        for (path, directories, filenames) in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)
                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)
                else:
                    paths_dict[install_path] = [file_path]
    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=package_files(data_files, ['models/', 'launch/', 'config/']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='tolomei.simone5@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_controller = rlg_quad_controller.inference_controller:main',
            'force_measure = rlg_quad_controller.force_measure:main',
            'trajectory_publisher = rlg_quad_controller.trajectory_publisher:main'
        ],
    },
)
