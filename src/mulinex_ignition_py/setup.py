from setuptools import find_packages, setup

package_name = 'mulinex_ignition_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        ],
    },
)
