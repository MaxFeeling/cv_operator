from setuptools import setup

setup(name="resnet_series",
        version="0.0.1",
        description="Resnets for prophet vision classification",
        author="for_carrots",
        author_email="songyexuan@4paradigm.com",
        install_requires=[
            #'tensorflow'
            'torch',
            "Pillow"
            # 'torchvision==0.6.1'
        ],
        python_requires='>=2.7',
        packages=["ResNet"
                  ],
        include_package_data = True,
        zip_safe=False
        )

