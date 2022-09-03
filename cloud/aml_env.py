r'''
Create aml environment with custom docker image built for dataset explorer

This is only meant to be called once for a new workspace to support, kept here more for record purpose
'''


import os

from azureml.core import Environment, Workspace

DOCKER_BASEIMAGE = 'oneocracr.azurecr.io/verticals/painter'
DOCKER_REGISTRY = 'oneocracr.azurecr.io'
DOCKER_USERNAME = 'oneocracr'
DOCKER_PASSWORD = os.environ.get('ONEOCRACR_PASSWORD')


if __name__ == '__main__':

    if not DOCKER_PASSWORD:
        raise RuntimeError('DOCKER_PASSWORD env not set')

    env = Environment('vdipainter')
    env.docker.base_image = DOCKER_BASEIMAGE
    env.docker.base_image_registry.address = DOCKER_REGISTRY
    env.docker.base_image_registry.username = DOCKER_USERNAME
    env.docker.base_image_registry.password = DOCKER_PASSWORD

    env.python.user_managed_dependencies = True
    env.python.interpreter_path = '/opt/conda/bin/python'

    ws = Workspace.from_config()
    env.register(ws)