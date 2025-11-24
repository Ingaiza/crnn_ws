from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'crnn_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    
    # packs the JSON with the code.
    package_data={
        'crnn_py': ['client_secrets.json'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ingaiza',
    maintainer_email='alvindavid898@gmail.com',
    description='CRNN Audio Classification Node',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'results = crnn_py.results:main',
            'upload = crnn_py.upload:main', 
        ],
    },
)