from setuptools import setup

package_name = 'quadcopter_cv'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'camera_processor = quadcopter_cv.camera_processor:main',
            'face_tracker = quadcopter_cv.face_tracker:main',
        ],
    },
)
