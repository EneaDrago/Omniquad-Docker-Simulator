from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'mulinex_ignition_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/rosbag_rec.launch.py']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*_launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='ros@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'getup = mulinex_ignition_py.getup:main',
            'gostraighton = mulinex_ignition_py.gostraighton:main',    
            'walking = mulinex_ignition_py.walking:main',          
            'move_wheels = mulinex_ignition_py.move_wheels:main',
            'move_pd = mulinex_ignition_py.move_pd:main',
            'move_wheels_2 = mulinex_ignition_py.move_wheels_2:main',
            'bridge_node = mulinex_ignition_py.bridge_node:main',
            'getup_omni_1 = mulinex_ignition_py.getup_omni_1:main',
        ],
    },
)
